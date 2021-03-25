#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Clock.h"

using namespace cv;

// PSF toute simple, faite "a la mano"
// le premier indice est le "k" du sous-échantillonnage
// le deuxième indice donne les coefficients par ligne ou colonne
// 0: -k/2  ---> k: k/2
const float moy[ 10 ][ 10 ] = {
  { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 },
  { 0.05, 0.08, 0.2, 0.34, 0.2, 0.08, 0.05, 0.0, 0.0, 0.0 },
  { 0.05, 0.08, 0.2, 0.34, 0.2, 0.08, 0.05, 0.0, 0.0, 0.0 },
  { 0.02, 0.04, 0.08, 0.18, 0.36, 0.18, 0.08, 0.04, 0.02, 0.0 },
  { 0.02, 0.04, 0.08, 0.18, 0.36, 0.18, 0.08, 0.04, 0.02, 0.0 }
};

Mat subsampleX( Mat g, int k )
{
  auto rows = g.rows;
  auto cols = g.cols;
  int ncols = cols / k;
  Mat gx( rows, ncols, CV_32FC1, 0.0f );
  //const int k_div_2 = k / 2;
  //const int lk      = 2 * k_div_2 + 1; 
  for ( int y = 0; y < rows; y++ )
    for ( int x = 0; x < ncols; x++ )
      {
        float sum_v = 0.0;
        float sum_c = 0.0;
        for ( int i = 0; i < k; ++i ) {
          sum_v += 1.0 * g.at< float >( y, k*x+i );
          sum_c += 1.0;
        }
        gx.at< float >( y, x ) = sum_v / sum_c;
      }
  // Mat output;
  // resize( gx, output, Size( cols, rows ) ); // linear interpolation by default
  return gx;
}

Mat subsampleY( Mat g, int k )
{
  auto rows = g.rows;
  auto cols = g.cols;
  int nrows = rows / k;
  Mat gy( nrows, cols, CV_32FC1, 0.0f );
  //const int k_div_2 = k / 2;
  //const int lk      = 2 * k_div_2 + 1; 
  for ( int y = 0; y < nrows; y++ )
    for ( int x = 0; x < cols; x++ )
      {
        float sum_v = 0.0;
        float sum_c = 0.0;
        for ( int i = 0; i < k; ++i ) {
          sum_v += 1.0 * g.at< float >( k*y+i, x );
          sum_c += 1.0;
        }
        gy.at< float >( y, x ) = sum_v / sum_c;
      }
  //Mat output;
  //resize( gy, output, Size( cols, rows ) ); // linear interpolation by default
  return gy;
}

Mat initU( Mat sx, Mat sy, int k )
{
  auto rows = sx.rows;
  auto cols = sy.cols;
  Mat u( rows, cols, CV_32FC1, 0.0f );
  Mat ix, iy;
  resize( sx, ix, Size( cols, rows ), 0.0, 0.0, INTER_NEAREST ); // linear interpolation by default
  resize( sy, iy, Size( cols, rows ), 0.0, 0.0, INTER_NEAREST ); // linear interpolation by default
  for ( int y = 0; y < rows; y++ )
    for ( int x = 0; x < cols; x++ )
      u.at< float >( y, x ) = 0.5 * ( ix.at< float >( y, x )
                                      + iy.at< float >( y, x ) );
  return u;
}

/// Initializes an AT process and returns images gx, gy, u and v.
std::tuple< Mat, Mat, Mat, Mat >
AT_create( Mat input, int k )
{
  auto rows = input.rows;
  auto cols = input.cols;
  int nrows = ( rows / k ) * k;
  int ncols = ( cols / k ) * k;
  Mat ninput = input( Rect( 0, 0, ncols, nrows ) );
  Mat g;
  ninput.convertTo( g, CV_32FC1, 1/255.0);
  Mat gx = subsampleX( g, k );
  Mat gy = subsampleY( g, k );
  Mat u  = initU( gx, gy, k );
  Mat v( rows+1, cols+1, CV_32FC1, 1.0f );
  return std::make_tuple( gx, gy, u, v );
}

inline float sqr( float x ) { return x*x; }


Mat AT_updateU( Mat u, Mat v, Mat gx, Mat gy, int k, float beta )
{
  //std::cout << "updateU" << std::endl;
  // v_(x,y) ---- vn ----- v_(x+1,y)
  //    |                      |
  //    vw     u_(x,y)         ve
  //    |                      |
  // v_(x,y+1) -- vs ----- v_(x+1,y+1)
    
  const auto rows = u.rows;
  const auto cols = u.cols;
  const int k_div_2 = k / 2;
  Mat gu( rows, cols, CV_32FC1, 0.0f );
  const auto rows_m_1 = rows-1;
  const auto cols_m_1 = cols-1;
  const float two_beta = 2.0 * beta;
  const float c = 1.0f / (float) k;
  const float diag_c   = sqrt( 2.0f ) / 2.0f;
  const float diag_beta = beta * diag_c;
  const int by = k; //1; //std::max( 1, k_div_2 );
  const int ey = rows - k; //rows_m_1; //std::min( rows_m_1, rows - k_div_2 );
  const int bx = k; //std::max( 1, k_div_2 );
  const int ex = cols - k; //cols_m_1; //std::min( cols_m_1, cols - k_div_2 );
  for ( int y = by; y < ey; y++ ) {
    const int yk = y / k;
    const int iy = y % k;
    for ( int x = bx; x < ex; x++ )
      {
        const int xk = x / k;
        const int ix = x % k;
        const float v_x_y     = v.at< float >( y, x );
        const float v_xp1_y   = v.at< float >( y, x+1 );
        const float v_x_yp1   = v.at< float >( y+1, x );
        const float v_xp1_yp1 = v.at< float >( y+1, x+1 );
        // if ( v_x_y != v_x_y ) {
        //   std::cerr << "NaN v_x_y value at " << x << "," << y 
        //             << std::endl;
        //   exit( 0 );
        // }
        // if ( v_xp1_y != v_xp1_y ) {
        //   std::cerr << "NaN v_xp1_y value at " << x+1 << "," << y 
        //             << std::endl;
        //   exit( 0 );
        // }
        // if ( v_x_yp1 != v_x_yp1 ) {
        //   std::cerr << "NaN v_x_yp1 value at " << x << "," << y+1
        //             << std::endl;
        //   exit( 0 );
        // }
        // if ( v_xp1_yp1 != v_xp1_yp1 ) {
        //   std::cerr << "NaN v_xp1_yp1 value at " << x+1 << "," << y+1
        //             << std::endl;
        //   exit( 0 );
        // }
        
        const float vw = 0.5*( v_x_y + v_x_yp1 );
        const float ve = 0.5*( v_xp1_y + v_xp1_yp1 );
        const float vn = 0.5*( v_x_y + v_xp1_y );
        const float vs = 0.5*( v_x_yp1 + v_xp1_yp1 );
        const float vw2 = sqr( vw );
        const float ve2 = sqr( ve );
        const float vn2 = sqr( vn );
        const float vs2 = sqr( vs );
        const float vnw2 = sqr( v_x_y );
        const float vne2 = sqr( v_xp1_y );
        const float vsw2 = sqr( v_x_yp1 );
        const float vse2 = sqr( v_xp1_yp1 );
        const float left = //4.0 * moy[ k ][ k_div_2 ]
          beta * ( ve2 + vw2 + vn2 + vs2 )
          + diag_beta * ( vnw2 + vne2 + vsw2 + vse2 );
        
        float  left_h = 0.0;
        float right_h = 0.0;
        for ( int i = 0; i < k; i++ ) {
          if ( i == ix ) left_h += c;
          else right_h -= c* u.at< float >( y, xk * k + i );
          if ( i == iy ) left_h += c;
          else right_h -= c* u.at< float >( yk * k + i, x );
        }
        const float right = 2.0 * ( gx.at< float >( y, xk )
                                    + gy.at< float >( yk, x ) )
          + beta * ( ve2 * ( u.at< float >( y, x+1 ) )
                     + vw2 * ( u.at< float >( y, x-1 ) )
                     + vn2 * ( u.at< float >( y-1, x ) )
                     + vs2 * ( u.at< float >( y+1, x ) ) )
          + diag_beta * ( vnw2 * u.at< float >( y-1, x-1 )
                          + vne2 * u.at< float >( y-1, x+1 )
                          + vsw2 * u.at< float >( y+1, x-1 )
                          + vse2 * u.at< float >( y+1, x+1 ) );
        // float right_h = 0.0;
        // for ( int i = -k_div_2; i <= k_div_2; ++i ) {
        //   if ( i != 0 ) {
        //     right_h -= moy[ k ][ i + k_div_2 ] * u.at< float >( y, x + i );
        //     right_h -= moy[ k ][ i + k_div_2 ] * u.at< float >( y + i, x );
        //   }
        // }
        const float val = ( right + 2.0f * right_h) / (left + 2.0 * left_h);
        // = std::min( 1.0f, std::max( 0.0f, (right + 2.0f * right_h) / left ) );
        gu.at< float >( y, x ) = val;
        // if ( val != val ) {
        //   std::cerr << "NaN value at " << x << "," << y 
        //             << " right=" << right
        //             << " 2*right_=" << 2.0f*right_h
        //             << " left=" << left
        //             << " 2*left_=" << 2.0f*left_h << std::endl;
        //   exit( 0 );
        // }
      }
  }
  return gu;
}


Mat AT_updateV( Mat u, Mat v, int k, float beta, float lambda, float epsilon )
{
  //std::cout << "updateV" << std::endl;
  // v_(x-1,y-1) ---- v_(x,y-1) ------ v_(x+1,y-1)
  //    |                 |                |
  //    |    u_(x-1,y-1)  |    u_(x,y-1)   |
  //    |                 |                |
  // v_(x-1,y) ------- v_(x,y) ------- v_(x+1,y)
  //    |                 |                |
  //    |     u_(x-1,y)   |     u_(x,y)    |
  //    |                 |                |
  // v_(x-1,y+1) ---- v_(x,y+1) ------ v_(x+1,y+1)
  const auto rows = v.rows;
  const auto cols = v.cols;
  Mat gv( rows, cols, CV_32FC1, 0.0f );
  const auto rows_m_1 = rows-1;
  const auto cols_m_1 = cols-1;
  const float half_b  = 0.5 * beta;
  const float  diag_c = sqrt(2.0) / 2.0; 
  const float left_le = 4.0 * lambda * epsilon * ( 1.0 + diag_c )
    + lambda / ( 2.0 * epsilon );
  const float right_le = 1.0 * lambda * epsilon;
  const float l_sur_2e = lambda / ( 2.0 * epsilon );
  const int by = k; //1; //std::max( 1, k_div_2 );
  const int ey = rows - k; //rows_m_1; //std::min( rows_m_1, rows - k_div_2 );
  const int bx = k; //std::max( 1, k_div_2 );
  const int ex = cols - k; //cols_m_1; //std::min( cols_m_1, cols - k_div_2 );
  for ( int y = by; y < ey; y++ ) 
    for ( int x = bx; x < ex; x++ )
      {
        const float u_xm1_ym1 = u.at< float >( y-1, x-1 );
        const float u_x_ym1   = u.at< float >( y-1, x );
        const float u_xm1_y   = u.at< float >( y, x-1 );
        const float u_x_y     = u.at< float >( y, x );
        const float ue = u_x_y - u_x_ym1;
        const float uw = u_xm1_y - u_xm1_ym1;
        const float un = u_x_ym1 - u_xm1_ym1;
        const float us = u_x_y - u_xm1_y;
        const float ue2 = sqr( ue ); 
        const float uw2 = sqr( uw ); 
        const float un2 = sqr( un ); 
        const float us2 = sqr( us );
        const float left= half_b * ( ue2 + uw2 + un2 + us2 ) + left_le;
        const float ve = v.at< float >( y, x+1 );
        const float vw = v.at< float >( y, x-1 );
        const float vn = v.at< float >( y-1, x );
        const float vs = v.at< float >( y+1, x );
        const float vnw = v.at< float >( y-1, x-1 );
        const float vne = v.at< float >( y-1, x+1 );
        const float vsw = v.at< float >( y+1, x-1 );
        const float vse = v.at< float >( y+1, x+1 );
        const float right =
          - half_b * ( ve * ue2 + vw * uw2 + vn * un2 + vs * us2 )
          + right_le * ( ve + vw + vn + vs
                       + diag_c * ( vne + vnw + vse + vsw ) )
          + l_sur_2e;
        gv.at< float >( y, x ) = right / left;
        // = std::min( 1.0f, std::max( 0.0f, right / left ) );
      }
  return gv;
}

float AT_optimizeU( Mat u, Mat v, Mat gx, Mat gy, int k,
                    float beta, float lambda, float epsilon, float gamma )
{
  Mat old_u = u.clone();
  Mat new_u = AT_updateU( u, v, gx, gy, k, beta );
  addWeighted( new_u, gamma, old_u, 1.0-gamma, 0.0, u);
  old_u -= u;
  // Mat gu = AT_gradU( u, v, g, beta );
  // Mat old_u = u.clone();
  // u -= gamma * gu;
  // old_u -= u;
  float norm = 0.0;
  int nb = 0;
  for ( auto it = old_u.begin<float>(), itE = old_u.end<float>(); it != itE; ++it ) {
    norm += (*it) * (*it);
    nb   += 1;
  }
  return sqrt( norm / (float) nb );
}

float AT_optimizeV( Mat u, Mat v, Mat gx, Mat gy, int k,
                    float beta, float lambda, float epsilon, float gamma )
{
  Mat old_v = v.clone();
  Mat new_v = AT_updateV( u, v, k, beta, lambda, epsilon );
  addWeighted( new_v, gamma, old_v, 1.0-gamma, 0.0, v);
  old_v -= v;
  float norm = 0.0;
  int nb = 0;
  for ( auto it = old_v.begin<float>(), itE = old_v.end<float>(); it != itE; ++it ) {
    norm += (*it) * (*it);
    nb   += 1;
  }
  return sqrt( norm / (float) nb );
}

void displaySize( std::string s, Mat A )
{
  std::cout << "Mat " << s << " is " << A.cols << "x" << A.rows << std::endl;
}

float
AT_optimizeUV( Mat u, Mat v, Mat gx, Mat gy, int k,
               float beta, float lambda, float epsilon,
               float gamma, int max_iter )
{
  float norm;
  float total_norm = 0.0;
  for ( int i = 0; i < max_iter; ++i )
    {
      norm = AT_optimizeU( u, v, gx, gy, k, beta, lambda, epsilon, gamma );
      // std::cout << "||u^(k+1)-u^(k)||_2 = " << norm << std::endl;
      if ( norm < 0.0001 ) break;
    }
  total_norm += norm;
  for ( int i = 0; i < max_iter; ++i )
    {
      norm = AT_optimizeV( u, v, gx, gy, k, beta, lambda, epsilon, gamma );
      // std::cout << "||v^(k+1)-v^(k)||_2 = " << norm << std::endl;
      if ( norm < 0.0001 ) break;
    }
  total_norm += norm;
  return total_norm;
}

int main( int argc, char* argv[] )
{
  if ( argc < 2 )
    {
      printf("usage: %s <image> <k>\n", argv[ 0 ]);
      return -1;
    }
  Mat gray_input;
  gray_input = imread( argv[1], IMREAD_GRAYSCALE ); //IMREAD_COLOR );
  if ( ! gray_input.data )
    {
      printf("No image data \n");
      return -1;
    }
  namedWindow("U", WINDOW_NORMAL ); //WINDOW_AUTOSIZE);
  namedWindow("V", WINDOW_NORMAL );
  int ibeta = 400;
  int ilambda = 1;
  int ilambda_100 = 00;
  int iepsilon1 = 800;
  int iepsilon2 = 25;
  int igamma = 50;
  int max_iter = 30;
  int k = argc >= 3 ? atoi( argv[ 2 ] ) : 4;
  createTrackbar("gamma (en 1/100)", "U", &igamma, 100, NULL );
  createTrackbar("max_iter", "U", &max_iter, 100, NULL );
  createTrackbar("beta (en %)", "U", &ibeta, 1000, NULL );
  createTrackbar("lambda (en %)", "V", &ilambda, 100, NULL );
  createTrackbar("lambda (en %%)", "V", &ilambda_100, 100, NULL );
  createTrackbar("epsilon1 (en %)", "V", &iepsilon1, 1000, NULL );
  createTrackbar("epsilon2 (en %)", "V", &iepsilon2, 500, NULL );
  Mat gx, gy, u, v;
  std::tie( gx, gy, u, v ) = AT_create( gray_input, k );
  displaySize( "u", u );
  displaySize( "v", v );
  displaySize( "gx", gx );
  displaySize( "gy", gy );
  float epsilon = iepsilon1 / 100.0;
  bool display = true;
  bool compute = false;
  Clock T;
  T.startClock();
  for(;;)
    {
      float beta = ibeta / 100.0;
      float lambda = ilambda / 100.0 + ilambda_100/ 10000.0;
      float epsilon1 = iepsilon1 / 100.0;
      float epsilon2 = iepsilon2 / 100.0;
      float gamma = igamma / 1000.0;
      int keycode = waitKey(100);
      char ascii  = keycode & 0xFF;  
      if ( ascii == 'q' ) break;
      else if ( ascii == 'd' ) {
        display = ! display;
      } else if ( ascii == 'c' ) {
        compute = true;
        epsilon = epsilon1;
        T.startClock();
      } else if ( ascii == 'i' )
        std::tie( gx, gy, u, v ) = AT_create( gray_input, k );
      if ( display ) {
        //Mat nu;
        //cv::normalize( u, nu, 0, 1, cv::NORM_MINMAX);
        //imshow("U", nu);
        imshow("U", u);
        imshow("V", v);
      }
      float norm  = 0.0;
      if ( compute ) {
        std::cout << "*********** epsilon = " << epsilon << std::endl;
        norm = AT_optimizeUV( u, v, gx, gy, k,
                              beta, lambda, epsilon, gamma, max_iter );
        std::cout << "     ||u^(k+1)-u^(k)||_2 + ||v^(k+1)-v^(k)||_2 = "
                  << norm << std::endl;
        if ( norm < 0.0001 && epsilon == epsilon2 ) {
          double t = T.stopClock();
          compute = false;
          std::cout << " ... in " << t << " ms." << std::endl;
          if ( ! display ) {
            imshow("U", u);
            imshow("V", v);
          }
        } else if ( norm < 0.0005 )
          epsilon = std::max( 7.0f * epsilon / 8.0f, epsilon2 );
      }
    }
  return 0;
}
