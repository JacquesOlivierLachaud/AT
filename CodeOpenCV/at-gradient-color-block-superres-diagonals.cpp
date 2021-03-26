#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Clock.h"

using namespace cv;

typedef float Real;
typedef Vec3f Value;
#define CVRealImageC1 CV_32FC1
#define CVRealImageC3 CV_32FC3
// typedef double Real;
// #define CVRealImageC1 CV_64FC1


Mat subsampleX( Mat g, int k )
{
  auto rows = g.rows;
  auto cols = g.cols;
  int ncols = cols / k;
  Mat gx( rows, ncols, CVRealImageC3, 0.0f );
  for ( int y = 0; y < rows; y++ )
    for ( int x = 0; x < ncols; x++ )
      {
        Value sum_v = 0.0;
        Real sum_c = 0.0;
        for ( int i = 0; i < k; ++i ) {
          sum_v += 1.0 * g.at< Value >( y, k*x+i );
          sum_c += 1.0;
        }
        gx.at< Value >( y, x ) = sum_v / sum_c;
      }
  return gx;
}

Mat subsampleY( Mat g, int k )
{
  auto rows = g.rows;
  auto cols = g.cols;
  int nrows = rows / k;
  Mat gy( nrows, cols, CVRealImageC3, 0.0f );
  //const int k_div_2 = k / 2;
  //const int lk      = 2 * k_div_2 + 1; 
  for ( int y = 0; y < nrows; y++ )
    for ( int x = 0; x < cols; x++ )
      {
        Value sum_v = { 0.0, 0.0, 0.0 };
        Real sum_c = 0.0;
        for ( int i = 0; i < k; ++i ) {
          sum_v += 1.0 * g.at< Value >( k*y+i, x );
          sum_c += 1.0;
        }
        gy.at< Value >( y, x ) = sum_v / sum_c;
      }
  //Mat output;
  //resize( gy, output, Size( cols, rows ) ); // linear interpolation by default
  return gy;
}

Mat initU( Mat sx, Mat sy, int k )
{
  auto rows = sx.rows;
  auto cols = sy.cols;
  Mat u( rows, cols, CVRealImageC3, 0.0f );
  Mat ix, iy;
  resize( sx, ix, Size( cols, rows ), 0.0, 0.0, INTER_NEAREST ); // linear interpolation by default
  resize( sy, iy, Size( cols, rows ), 0.0, 0.0, INTER_NEAREST ); // linear interpolation by default
  for ( int y = 0; y < rows; y++ )
    for ( int x = 0; x < cols; x++ )
      u.at< Value >( y, x ) = 0.5 * ( ix.at< Value >( y, x )
                                      + iy.at< Value >( y, x ) );
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
  ninput.convertTo( g, CVRealImageC3, 1/255.0);
  Mat gx = subsampleX( g, k );
  Mat gy = subsampleY( g, k );
  Mat u  = initU( gx, gy, k );
  Mat v( rows+1, cols+1, CVRealImageC1, 1.0f );
  return std::make_tuple( gx, gy, u, v );
}

inline Real sqr( Real x )  { return x*x; }
inline Real sqr( Value x ) { return sqr( x[ 0 ] ) + sqr( x[ 1 ] ) + sqr( x[ 2 ] ); }


Mat AT_updateU( Mat u, Mat v, Mat gx, Mat gy, int k, Real beta )
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
  const Value zero = { 0.0f, 0.0f, 0.0f };
  Mat gu( rows, cols, CVRealImageC3, zero );
  const auto rows_m_1 = rows-1;
  const auto cols_m_1 = cols-1;
  const Real two_beta = 2.0 * beta;
  const Real c = 1.0f / (Real) k;
  const Real diag_c   = sqrt( 2.0f ) / 2.0f;
  const Real diag_beta = beta * diag_c;
  const int by = 1; //k
  const int ey = rows - 1; //rows - k
  const int bx = 1; //k
  const int ex = cols - 1; //cols - k
  for ( int y = by; y < ey; y++ ) {
    const int yk = y / k;
    const int iy = y % k;
    for ( int x = bx; x < ex; x++ )
      {
        const int xk = x / k;
        const int ix = x % k;
        const Real v_x_y     = v.at< Real >( y, x );
        const Real v_xp1_y   = v.at< Real >( y, x+1 );
        const Real v_x_yp1   = v.at< Real >( y+1, x );
        const Real v_xp1_yp1 = v.at< Real >( y+1, x+1 );
        
        const Real vw = 0.5*( v_x_y + v_x_yp1 );
        const Real ve = 0.5*( v_xp1_y + v_xp1_yp1 );
        const Real vn = 0.5*( v_x_y + v_xp1_y );
        const Real vs = 0.5*( v_x_yp1 + v_xp1_yp1 );
        const Real vw2 = sqr( vw );
        const Real ve2 = sqr( ve );
        const Real vn2 = sqr( vn );
        const Real vs2 = sqr( vs );
        const Real vnw2 = sqr( v_x_y );
        const Real vne2 = sqr( v_xp1_y );
        const Real vsw2 = sqr( v_x_yp1 );
        const Real vse2 = sqr( v_xp1_yp1 );
        const Real left = beta * ( ve2 + vw2 + vn2 + vs2 )
          + diag_beta * ( vnw2 + vne2 + vsw2 + vse2 );
        
        Real   left_h = 0.0;
        Value right_h = 0.0;
        for ( int i = 0; i < k; i++ ) {
          if ( i == ix ) left_h += c;
          else right_h -= c* u.at< Value >( y, xk * k + i );
          if ( i == iy ) left_h += c;
          else right_h -= c* u.at< Value >( yk * k + i, x );
        }
        const Value right = 2.0 * ( gx.at< Value >( y, xk )
                                    + gy.at< Value >( yk, x ) )
          + beta * ( ve2 * ( u.at< Value >( y, x+1 ) )
                     + vw2 * ( u.at< Value >( y, x-1 ) )
                     + vn2 * ( u.at< Value >( y-1, x ) )
                     + vs2 * ( u.at< Value >( y+1, x ) ) )
          + diag_beta * ( vnw2 * u.at< Value >( y-1, x-1 )
                          + vne2 * u.at< Value >( y-1, x+1 )
                          + vsw2 * u.at< Value >( y+1, x-1 )
                          + vse2 * u.at< Value >( y+1, x+1 ) );
        const Value val = ( right + 2.0f * right_h) / (left + 2.0 * left_h);
        gu.at< Value >( y, x ) = val;
      }
  }
  return gu;
}


Mat AT_updateV( Mat u, Mat v, int k, Real beta, Real lambda, Real epsilon )
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
  Mat gv( rows, cols, CVRealImageC1, 0.0f );
  const auto rows_m_1 = rows-1;
  const auto cols_m_1 = cols-1;
  const Real half_b  = 0.5 * beta;
  const Real  diag_c = sqrt(2.0) / 2.0; 
  const Real left_le = 4.0 * lambda * epsilon * ( 1.0 + diag_c )
    + lambda / ( 2.0 * epsilon );
  const Real right_le = 1.0 * lambda * epsilon;
  const Real l_sur_2e = lambda / ( 2.0 * epsilon );
  const int by = k; //k
  const int ey = rows - k; //rows - k
  const int bx = k; //k
  const int ex = cols - k; //cols - k
  for ( int y = by; y < ey; y++ ) 
    for ( int x = bx; x < ex; x++ )
      {
        const Value u_xm1_ym1 = u.at< Value >( y-1, x-1 );
        const Value u_x_ym1   = u.at< Value >( y-1, x );
        const Value u_xm1_y   = u.at< Value >( y, x-1 );
        const Value u_x_y     = u.at< Value >( y, x );
        const Value ue = u_x_y - u_x_ym1;
        const Value uw = u_xm1_y - u_xm1_ym1;
        const Value un = u_x_ym1 - u_xm1_ym1;
        const Value us = u_x_y - u_xm1_y;
        const Real ue2 = sqr( ue ); 
        const Real uw2 = sqr( uw ); 
        const Real un2 = sqr( un ); 
        const Real us2 = sqr( us );
        const Real left= half_b * ( ue2 + uw2 + un2 + us2 ) + left_le;
        const Real ve = v.at< Real >( y, x+1 );
        const Real vw = v.at< Real >( y, x-1 );
        const Real vn = v.at< Real >( y-1, x );
        const Real vs = v.at< Real >( y+1, x );
        const Real vnw = v.at< Real >( y-1, x-1 );
        const Real vne = v.at< Real >( y-1, x+1 );
        const Real vsw = v.at< Real >( y+1, x-1 );
        const Real vse = v.at< Real >( y+1, x+1 );
        const Real right =
          - half_b * ( ve * ue2 + vw * uw2 + vn * un2 + vs * us2 )
          + right_le * ( ve + vw + vn + vs
                       + diag_c * ( vne + vnw + vse + vsw ) )
          + l_sur_2e;
        gv.at< Real >( y, x ) = right / left;
      }
  return gv;
}

Real AT_optimizeU( Mat u, Mat v, Mat gx, Mat gy, int k,
                    Real beta, Real lambda, Real epsilon, Real gamma )
{
  Mat old_u = u.clone();
  //std::cout << "UpdateU" << std::endl;
  Mat new_u = AT_updateU( u, v, gx, gy, k, beta );
  //std::cout << "addWeighted" << std::endl;
  Value zero { 0.0, 0.0, 0.0 };
  addWeighted( new_u, gamma, old_u, 1.0-gamma, 0.0, u);
  old_u -= u;
  Real norm = 0.0;
  int nb = 0;
  //std::cout << "ComputeNormDu" << std::endl;
  for ( auto it = old_u.begin<Value>(), itE = old_u.end<Value>(); it != itE; ++it ) {
    norm += sqr( *it );
    nb   += 1;
  }
  return sqrt( norm / (Real) (3.0f*nb) );
}

Real AT_optimizeV( Mat u, Mat v, Mat gx, Mat gy, int k,
                    Real beta, Real lambda, Real epsilon, Real gamma )
{
  Mat old_v = v.clone();
  Mat new_v = AT_updateV( u, v, k, beta, lambda, epsilon );
  addWeighted( new_v, gamma, old_v, 1.0-gamma, 0.0, v);
  old_v -= v;
  Real norm = 0.0;
  int nb = 0;
  for ( auto it = old_v.begin<Real>(), itE = old_v.end<Real>(); it != itE; ++it ) {
    norm += (*it) * (*it);
    nb   += 1;
  }
  return sqrt( norm / (Real) nb );
}

void displaySize( std::string s, Mat A )
{
  std::cout << "Mat " << s << " is " << A.cols << "x" << A.rows << std::endl;
}

Real
AT_optimizeUV( Mat u, Mat v, Mat gx, Mat gy, int k,
               Real beta, Real lambda, Real epsilon,
               Real gamma, int max_iter )
{
  Real norm;
  Real total_norm = 0.0;
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
  Mat color_input;
  color_input = imread( argv[1], IMREAD_COLOR );
  if ( ! color_input.data )
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
  createTrackbar("epsilon1 (en %)", "V", &iepsilon1, 2000, NULL );
  createTrackbar("epsilon2 (en %)", "V", &iepsilon2, 500, NULL );
  Mat gx, gy, u, v;
  std::tie( gx, gy, u, v ) = AT_create( color_input, k );
  displaySize( "u", u );
  displaySize( "v", v );
  displaySize( "gx", gx );
  displaySize( "gy", gy );
  Real epsilon = iepsilon1 / 100.0;
  bool display = true;
  bool compute = false;
  Clock T;
  T.startClock();
  for(;;)
    {
      Real beta = ibeta / 100.0;
      Real lambda = ilambda / 100.0 + ilambda_100/ 10000.0;
      Real epsilon1 = iepsilon1 / 100.0;
      Real epsilon2 = iepsilon2 / 100.0;
      Real gamma = igamma / 1000.0;
      int keycode = waitKey(100);
      char ascii  = keycode & 0xFF;  
      if ( ascii == 'q' ) break;
      else if ( ascii == 'd' ) {
        display = ! display;
      } else if ( ascii == 'c' ) {
        if ( compute == false ) {
          compute = true;
          epsilon = epsilon1;
          T.startClock();
        } else {
          double t = T.stopClock();
          compute = false;
          std::cout << " ... in " << t << " ms." << std::endl;
        }
      } else if ( ascii == 'i' ) {
        double t = T.stopClock();
        compute = false;
        std::cout << " ... in " << t << " ms." << std::endl;
        std::tie( gx, gy, u, v ) = AT_create( color_input, k );
        imshow("U", u);
        imshow("V", v);
      }
      if ( display ) {
        //Mat nu;
        //cv::normalize( u, nu, 0, 1, cv::NORM_MINMAX);
        //imshow("U", nu);
        imshow("U", u);
        imshow("V", v);
      }
      Real norm  = 0.0;
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
