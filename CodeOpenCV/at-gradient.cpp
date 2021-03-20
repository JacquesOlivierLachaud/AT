#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Clock.h"

using namespace cv;

/// Initializes an AT process and returns images g, u and v.
std::tuple< Mat, Mat, Mat >
AT_create( Mat input )
{
  auto rows = input.rows;
  auto cols = input.cols;
  Mat g;
  input.convertTo( g, CV_32FC1, 1/255.0);
  Mat u = g.clone();
  Mat v( rows+1, cols+1, CV_32FC1, 1.0f );
  return std::make_tuple( g, u, v );
}

inline float sqr( float x ) { return x*x; }

Mat AT_gradU( Mat u, Mat v, Mat g, float beta )
{
  // v_(x,y) ---- vn ----- v_(x+1,y)
  //    |                      |
  //    vw     u_(x,y)         ve
  //    |                      |
  // v_(x,y+1) -- vs ----- v_(x+1,y+1)
  const auto rows = u.rows;
  const auto cols = u.cols;
  Mat gu( rows, cols, CV_32FC1, 0.0f );
  for ( int y = 1; y < rows-1; y++ )
    for ( int x = 1; x < cols-1; x++ )
      {
        float vw = 0.5*( v.at< float >( y, x ) + v.at< float >( y+1, x ) ); 
        float ve = 0.5*( v.at< float >( y, x+1 ) + v.at< float >( y+1, x+1 ) ); 
        float vn = 0.5*( v.at< float >( y, x ) + v.at< float >( y, x+1 ) ); 
        float vs = 0.5*( v.at< float >( y+1, x ) + v.at< float >( y+1, x+1 ) );
        float vw2 = sqr( vw );
        float ve2 = sqr( ve );
        float vn2 = sqr( vn );
        float vs2 = sqr( vs );
        float uxy = u.at< float >( y, x );
        float gg = 2.0 * ( uxy - g.at< float >( y, x ) )
          - 2.0 * beta * ( ve2 * ( u.at< float >( y, x+1 ) - uxy )
                           + vw2 * ( u.at< float >( y, x-1 ) - uxy )
                           + vn2 * ( u.at< float >( y-1, x ) - uxy )
                           + vs2 * ( u.at< float >( y+1, x ) - uxy ) );
        gu.at< float >( y, x ) = gg;
      }
  return gu;
}

Mat AT_updateU( Mat u, Mat v, Mat g, float beta )
{
  // v_(x,y) ---- vn ----- v_(x+1,y)
  //    |                      |
  //    vw     u_(x,y)         ve
  //    |                      |
  // v_(x,y+1) -- vs ----- v_(x+1,y+1)
  const auto rows = u.rows;
  const auto cols = u.cols;
  Mat gu( rows, cols, CV_32FC1, 0.0f );
  for ( int y = 1; y < rows-1; y++ )
    for ( int x = 1; x < cols-1; x++ )
      {
        float vw = 0.5*( v.at< float >( y, x ) + v.at< float >( y+1, x ) ); 
        float ve = 0.5*( v.at< float >( y, x+1 ) + v.at< float >( y+1, x+1 ) ); 
        float vn = 0.5*( v.at< float >( y, x ) + v.at< float >( y, x+1 ) ); 
        float vs = 0.5*( v.at< float >( y+1, x ) + v.at< float >( y+1, x+1 ) );
        float vw2 = sqr( vw );
        float ve2 = sqr( ve );
        float vn2 = sqr( vn );
        float vs2 = sqr( vs );
        float uxy = u.at< float >( y, x );
        float left = 2.0 * (1.0 + beta * ( ve2 + vw2 + vn2 + vs2 ) );
        
        float right = 2.0 * ( g.at< float >( y, x ) )
          + 2.0 * beta * ( ve2 * ( u.at< float >( y, x+1 ) )
                           + vw2 * ( u.at< float >( y, x-1 ) )
                           + vn2 * ( u.at< float >( y-1, x ) )
                           + vs2 * ( u.at< float >( y+1, x ) ) );
        float nuxy = right / left;
        gu.at< float >( y, x ) = nuxy;
      }
  return gu;
}

Mat AT_gradV( Mat u, Mat v, Mat g, float beta, float lambda, float epsilon )
{
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
  for ( int y = 1; y < rows-1; y++ )
    for ( int x = 1; x < cols-1; x++ )
      {
        float ue = ( u.at< float >( y, x ) - u.at< float >( y-1, x ) ); 
        float uw = ( u.at< float >( y, x-1 ) - u.at< float >( y-1, x-1 ) ); 
        float un = ( u.at< float >( y-1, x ) - u.at< float >( y-1, x-1 ) ); 
        float us = ( u.at< float >( y, x ) - u.at< float >( y, x-1 ) ); 
        float ue2 = sqr( ue ); 
        float uw2 = sqr( uw ); 
        float un2 = sqr( un ); 
        float us2 = sqr( us );
        float gg = v.at< float >( y, x ) * ( 0.5 * beta * ( ue2 + uw2 + un2 + us2 )
                                             - 8.0 * lambda * epsilon
                                             + lambda / ( 2.0 * epsilon ) )
          + 0.5 * beta * ( v.at< float >( y, x+1 ) * ue2
                           + v.at< float >( y, x-1 ) * uw2 
                           + v.at< float >( y-1, x ) * un2 
                           + v.at< float >( y+1, x ) * us2 )
          + 2.0 * lambda * epsilon * ( v.at< float >( y, x+1 )
                                       + v.at< float >( y, x-1 )
                                       + v.at< float >( y-1, x )
                                       + v.at< float >( y+1, x ) )
          - lambda  / ( 2.0 * epsilon );
        gv.at< float >( y, x ) = gg;
      }
  return gv;
}

Mat AT_updateV( Mat u, Mat v, Mat g, float beta, float lambda, float epsilon )
{
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
  const float left_le = 8.0 * lambda * epsilon + lambda / ( 2.0 * epsilon );
  const float  two_le = 2.0 * lambda * epsilon;
  const float l_sur_2e = lambda / ( 2.0 * epsilon );
  for ( int y = 1; y < rows_m_1; y++ )
    for ( int x = 1; x < cols_m_1; x++ )
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
        const float right = -half_b * ( ve * ue2 + vw * uw2 + vn * un2 + vs * us2 )
                             + two_le * ( ve + vw + vn + vs )
                             + l_sur_2e;
        gv.at< float >( y, x ) = right / left;
      }
  return gv;
}

float AT_optimizeU( Mat u, Mat v, Mat g, float beta, float lambda, float epsilon, float gamma )
{
  Mat old_u = u.clone();
  Mat new_u = AT_updateU( u, v, g, beta );
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

float AT_optimizeV( Mat u, Mat v, Mat g, float beta, float lambda, float epsilon, float gamma )
{
  Mat old_v = v.clone();
  Mat new_v = AT_updateV( u, v, g, beta, lambda, epsilon );
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

float
AT_optimizeUV( Mat u, Mat v, Mat g,
               float beta, float lambda, float epsilon,
               float gamma, int max_iter )
{
  float norm;
  float total_norm = 0.0;
  for ( int i = 0; i < max_iter; ++i )
    {
      norm = AT_optimizeU( u, v, g, beta, lambda, epsilon, gamma );
      // std::cout << "||u^(k+1)-u^(k)||_2 = " << norm << std::endl;
      if ( norm < 0.0001 ) break;
    }
  total_norm += norm;
  for ( int i = 0; i < max_iter; ++i )
    {
      norm = AT_optimizeV( u, v, g, beta, lambda, epsilon, gamma );
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
      printf("usage: %s image beta lambda epsilon gamma\n", argv[ 0 ]);
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
  int ibeta = 50;
  int ilambda = 5;
  int ilambda_100 = 0;
  int iepsilon1 = 400;
  int iepsilon2 = 50;
  int igamma = 50;
  int max_iter = 20;
  createTrackbar("gamma (en %)", "U", &igamma, 100, NULL );
  createTrackbar("max_iter", "U", &max_iter, 100, NULL );
  createTrackbar("beta (en %)", "U", &ibeta, 1000, NULL );
  createTrackbar("lambda (en %)", "V", &ilambda, 100, NULL );
  createTrackbar("lambda (en %%)", "V", &ilambda_100, 100, NULL );
  createTrackbar("epsilon1 (en %)", "V", &iepsilon1, 500, NULL );
  createTrackbar("epsilon2 (en %)", "V", &iepsilon2, 500, NULL );
  Mat g, u, v;
  std::tie( g, u, v ) = AT_create( gray_input );
  float epsilon = iepsilon1 / 100.0;
  bool display = true;
  bool compute = true;
  Clock T;
  T.startClock();
  for(;;)
    {
      float beta = ibeta / 100.0;
      float lambda = ilambda / 100.0 + ilambda_100/ 10000.0;
      float epsilon1 = iepsilon1 / 100.0;
      float epsilon2 = iepsilon2 / 100.0;
      float gamma = igamma / 100.0;
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
        std::tie( g, u, v ) = AT_create( gray_input );
      if ( display ) {
        imshow("U", u);
        imshow("V", v);
      }
      float norm  = 0.0;
      if ( compute ) {
        std::cout << "*********** epsilon = " << epsilon << std::endl;
        norm = AT_optimizeUV( u, v, g, beta, lambda, epsilon, gamma, max_iter );
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
        } else if ( norm < 0.001 )
          epsilon = std::max( 0.75f * epsilon, epsilon2 );
      }
    }
  return 0;
}
