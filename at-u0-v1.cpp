#include <sstream>
#include <string>
#include <boost/format.hpp>
//#include <boost/regex.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// always include EigenSupport.h before any other Eigen headers
#include "DGtal/math/linalg/EigenDecomposition.h"

#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

#include "DGtal/base/Common.h"
#include "DGtal/helpers/StdDefs.h"
#include "DGtal/io/readers/GenericReader.h"
#include "DGtal/io/writers/GenericWriter.h"
#include "DGtal/io/boards/Board2D.h"
#include "DGtal/math/linalg/EigenSupport.h"
#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"

// RealFFT
#include <DGtal/kernel/domains/HyperRectDomain.h>
#include <DGtal/kernel/SpaceND.h>
#include <DGtal/images/ImageContainerBySTLVector.h>
#include "VTKWriter.h"

// StructureTensor
#include "structureTensor.h"


using namespace std;
using namespace DGtal;
using namespace Eigen;

double standard_deviation(const VectorXd& xx)
{
  const double mean = xx.mean();
  double accum = 0;
  for (int kk=0, kk_max=xx.rows(); kk<kk_max; kk++)
    accum += (xx(kk)-mean)*(xx(kk)-mean);
  return sqrt(accum)/(xx.size()-1);
}

template <typename Calculus, typename Image>
void PrimalForm0ToImage( const Calculus& calculus, const typename Calculus::PrimalForm0& u, Image& image )
{
  double min_u = u.myContainer[ 0 ];
  double max_u = u.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      min_u = min( min_u, u.myContainer[ index ] );
      max_u = max( max_u, u.myContainer[ index ] );
    }
  trace.info() << "min_u=" << min_u << " max_u=" << max_u << std::endl;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = u.getSCell( index );
      //int g = (int) round( ( u.myContainer[ index ] - min_u ) * 255.0 /( max_u - min_u ) );
      int g = (int) round( u.myContainer[ index ] * 255.0 );
      g = std::max( 0 , std::min( 255, g ) );
      image.setValue( calculus.myKSpace.sCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void PrimalForm1ToImage( const Calculus& calculus, const typename Calculus::PrimalForm1& v, Image& image )
{
  double min_v = v.myContainer[ 0 ];
  double max_v = v.myContainer[ 0 ];
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      min_v = min( min_v, v.myContainer[ index ] );
      max_v = max( max_v, v.myContainer[ index ] );
    }
  trace.info() << "min_v=" << min_v << " max_v=" << max_v << std::endl;
  for ( typename Image::Iterator it = image.begin(), itE = image.end(); it != itE; ++it )
    *it = 255;
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {
      const typename Calculus::SCell& cell = v.getSCell( index );
      //int g = (int) round( ( v.myContainer[ index ] - min_v ) * 255.0 / ( max_v - min_v ) );
      int g = (int) round( v.myContainer[ index ] * 255.0 );
      g = std::max( 0 , std::min( 255, g ) );
      image.setValue( calculus.myKSpace.sKCoords( cell ), g );
    }
}

template <typename Calculus, typename Image>
void savePrimalForm0ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm0& u, const string& filename )
{
    Image end_image = image;
    PrimalForm0ToImage( calculus, u, end_image );
    ostringstream ossU;
    ossU << filename;
    string str_image_u = ossU.str();
    end_image >> str_image_u.c_str();
}

template <typename Calculus, typename Image>
void savePrimalForm1ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm1& v, const string& filename )
{
    Image end_image = image;
    PrimalForm1ToImage( calculus,v, end_image );
    ostringstream ossV;
    ossV << filename;
    string str_image_v = ossV.str();
    end_image >> str_image_v.c_str();
}

double tronc( const double& nb, const int& p )
{
  int i = pow(10,p) * nb;
  return i/pow(10,p);
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 diag( const Calculus& calculus,
                                         const typename Calculus::PrimalForm0& v )
{
  typename Calculus::PrimalIdentity0 diag_v = calculus.template identity<0, PRIMAL>();
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++ )
    diag_v.myContainer.coeffRef( index, index ) = v.myContainer( index );
  return diag_v;
}

template <typename Calculus>
typename Calculus::PrimalIdentity1 diag( const Calculus& calculus,
                                         const typename Calculus::PrimalForm1& v )
{
  typename Calculus::PrimalIdentity1 diag_v = calculus.template identity<1, PRIMAL>();
  for ( typename Calculus::Index index = 0; index < v.myContainer.rows(); index++ )
    diag_v.myContainer.coeffRef( index, index ) = v.myContainer( index );
  return diag_v;
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 square( const Calculus& calculus,
                                           const typename Calculus::PrimalIdentity0& B )
{
  typename Calculus::PrimalIdentity0 tB_B = calculus.template identity<0, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
typename Calculus::PrimalIdentity0 square( const Calculus& calculus,
                                           const typename Calculus::PrimalDerivative0& B )
{
  typename Calculus::PrimalIdentity0 tB_B = calculus.template identity<0, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
typename Calculus::PrimalIdentity1 square( const Calculus& calculus,
                                           const typename Calculus::PrimalIdentity1& B )
{
  typename Calculus::PrimalIdentity1 tB_B = calculus.template identity<1, PRIMAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm0& u,
                     const typename Calculus::PrimalForm0& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm1& u,
                     const typename Calculus::PrimalForm1& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::PrimalForm2& u,
                     const typename Calculus::PrimalForm2& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus, typename ImageDouble>
double interpolation(const Calculus& calculus, const ImageDouble& I, int const& i, int const& j, double const& x, double const& y)
//double interpolation(const ImageDouble& I, typename Calculus::SCell const& cell, RealVector const& n)
{

    typedef typename Calculus::KSpace KSpace;
    typedef typename KSpace::Point Point;

    const KSpace& K = calculus.myKSpace;
    const int n  = K.size(0); // = w-1
    const int m  = K.size(1); // = h-1

    // construction of the square {P0, P1, P2, P3} which contains (i+x,j+y) s.t.
    // P0 = (i,j), P1 = (i+/-1,j), P2 = (i,j+/-1), P3 = (i+/-1,j+/-1)
    int i1 = 0, i2 = 0, i3 = 0;
    int j1 = 0, j2 = 0, j3 = 0;

    // min(n+1, max(...)) ????
    // TODO : finish boundary conditions
    if( x >= 0 && y >= 0)
    {
        i1 = std::min(n, std::max(0,i+1));  j1 = j;                             //const Point P1 (i+1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j+1));  //const Point P2 (i  ,j-1);
        i3 = std::min(n, std::max(0,i+1));  j3 = std::min(m, std::max(0,j+1));  //const Point P3 (i+1,j-1);
    }
    else if( x >= 0 && y < 0)
    {
        i1 = std::min(n, std::max(0,i+1));  j1 = j;                             //const Point P1 (i+1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j-1));  //const Point P2 (i  ,j+1);
        i3 = std::min(n, std::max(0,i+1));  j3 = std::min(m, std::max(0,j-1));  //const Point P3 (i+1,j+1);
    }
    else if( x < 0 && y >= 0)
    {
        i1 = std::min(n, std::max(0,i-1));  j1 = j;                             //const Point P1 (i-1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j+1));  //const Point P2 (i  ,j-1);
        i3 = std::min(n, std::max(0,i-1));  j3 = std::min(m, std::max(0,j+1));  //const Point P3 (i-1,j-1);
    }
    else if( x < 0 && y < 0)
    {
        i1 = std::min(n, std::max(0,i-1));  j1 = j;                             //const Point P1 (i-1,j  );
        i2 = i;                             j2 = std::min(m, std::max(0,j-1));  //const Point P2 (i  ,j+1);
        i3 = std::min(n, std::max(0,i-1));  j3 = std::min(m, std::max(0,j-1));  //const Point P3 (i-1,j+1);
    }

    const Point P0 (i ,j );
    const Point P1 (i1,j1);
    const Point P2 (i2,j2);
    const Point P3 (i3,j3);


//    std::cout << " (i,j)= ("      << i << "," << j << ")"
//              << " (x,y)= ("      << x << "," << y << ")"
//              << " P(i,j)= "      << P0
//              << " P(i+1,j)= "    << P1
//              << " P(i,j+1)= "    << P2
//              << " P(i+1,j+1)= "  << P3
//              << std::endl;


    const double a = std::fabs(x);
    const double b = std::fabs(y);

    double val =   (1.0-a)*(1.0-b) * I(P0)
                 +    a   *(1.0-b) * I(P1)
                 + (1.0-a)*   b * I(P2)
                 +    a   *   b    * I(P3);

    return val;

}


namespace DGtal {
  template <typename TComponent, DGtal::Dimension TM, DGtal::Dimension TN>
  bool
  operator!=( const DGtal::SimpleMatrix< TComponent, TM, TN >& m1,
          const DGtal::SimpleMatrix< TComponent, TM, TN >& m2 )
  {
    return ! ( m1.operator==( m2 ) );
  }
}

int main( int argc, char* argv[] )
{
  using namespace Z2i;
  typedef ImageContainerBySTLVector<Domain, unsigned char>      Image;
  typedef ImageContainerBySTLVector<Domain, double>             ImageDouble;
  typedef ImageContainerBySTLVector<Domain,
                    SimpleMatrix<double,2,2> >  ImageSimpleMatrix2d;
  typedef ImageContainerBySTLVector<Domain, RealVector>         ImageVector;

  typedef std::vector< unsigned char >::iterator 			ImageIterator;
  typedef std::vector< double >::iterator 					ImageDoubleIterator;
  typedef std::vector< SimpleMatrix<double,2,2> >::iterator ImageSimpleMatrix2dIterator;
  typedef std::vector< RealVector >::iterator 				ImageVectorIterator;

  // parse command line ----------------------------------------------
  namespace po = boost::program_options;
  po::options_description general_opt("Allowed options are: ");
  general_opt.add_options()
    ("help,h", "display this message")
    ("input,i", po::value<string>(), "the input image filename." )
    ("output,o", po::value<string>()->default_value( "AT" ), "the output image basename." )
    ("lambda,l", po::value<double>(), "the parameter lambda." )
    ("lambda-1,1", po::value<double>()->default_value( 0.3125 ), "the initial parameter lambda (l1)." ) // 0.3125
    ("lambda-2,2", po::value<double>()->default_value( 0.00005 ), "the final parameter lambda (l2)." )
    ("lambda-ratio,q", po::value<double>()->default_value( sqrt(2) ), "the division ratio for lambda from l1 to l2." )
    ("alpha,a", po::value<double>()->default_value( 1.0 ), "the parameter alpha." )
    ("epsilon,e", po::value<double>()->default_value( 4.0/64.0 ), "the initial parameter epsilon." )
    //("gridstep,g", po::value<double>()->default_value( 1.0 ), "the parameter h, i.e. the gridstep." )
    ("nbiter,n", po::value<int>()->default_value( 10 ), "the maximum number of iterations." )
    ("sigma,s", po::value<double>()->default_value( 2.0 ), "the parameter of the first convolution." )
    ("rho,r", po::value<double>()->default_value( 3.0 ), "the parameter of the second convolution." )
    ("image-size,t", po::value<double>()->default_value( 64.0 ), "the size of the image." )
    ;

  bool parseOK=true;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, general_opt), vm);
  } catch ( const exception& ex ) {
    parseOK = false;
    cerr << "Error checking program options: "<< ex.what()<< endl;
  }
  po::notify(vm);
  if ( ! parseOK || vm.count("help") || !vm.count("input") )
    {
      cerr << "Usage: " << argv[0] << " -i toto.pgm\n"
       << "Computes the Ambrosio-Tortorelli reconstruction/segmentation of an input image."
       << endl << endl
       << " / "
       << endl
       << " | a.(u-g)^2 + v^2 |grad u|^2 + le.|grad v|^2 + (l/4e).(1-v)^2 "
       << endl
       << " / "
       << endl << endl
       << general_opt << "\n";
      return 1;
    }
  string f1 = vm[ "input" ].as<string>();
  string f2 = vm[ "output" ].as<string>();
  double l1  = vm[ "lambda-1" ].as<double>();
  double l2  = vm[ "lambda-2" ].as<double>();
  double lr  = vm[ "lambda-ratio" ].as<double>();
  if ( vm.count( "lambda" ) ) l1 = l2 = vm[ "lambda" ].as<double>();
  if ( l2 > l1 ) l2 = l1;
  if ( lr <= 1.0 ) lr = sqrt(2);
  double a  = vm[ "alpha" ].as<double>();
  double e  = vm[ "epsilon" ].as<double>();
  double t  = vm[ "image-size" ].as<double>();
  //double h  = vm[ "gridstep" ].as<double>();
  double h  = 1.0 / t;

  int    n  = vm[ "nbiter" ].as<int>();
  double s  = vm[ "sigma" ].as<double>();
  double r  = vm[ "rho" ].as<double>();

  trace.beginBlock("Reading image");
  Image image = GenericReader<Image>::import( f1 );
  Image end_image = image;
  trace.endBlock();

  // opening file
  const string file = f2 + ".txt";
  ofstream f(file.c_str());
  f << "#  l \t" << " a \t" << " e \t" << "a(u-g)^2 \t" << "v^2|grad u|^2 \t" << "  le|grad v|^2 \t" << "  l(1-v)^2/4e \t" << " l.per \t" << "AT tot"<< endl;

  trace.beginBlock("Creating calculus");
  typedef DiscreteExteriorCalculus<2,2, EigenLinearAlgebraBackend> Calculus;
  Domain domain = image.domain();
  Point p0 = domain.lowerBound(); p0 *= 2;
  Point p1 = domain.upperBound(); p1 *= 2;
  Domain kdomain( p0, p1 );
  Image dbl_image( kdomain );
  Calculus calculus;
  calculus.initKSpace( ConstAlias<Domain>( domain ) );
  const KSpace& K = calculus.myKSpace;
  // Les pixels sont des 0-cellules du primal.
  for ( Domain::ConstIterator it = kdomain.begin(), itE = kdomain.end(); it != itE; ++it )
    calculus.insertSCell( K.sCell( *it ) ); // ajoute toutes les cellules de Khalimsky.
  calculus.updateIndexes();
  trace.info() << calculus << endl;
  Calculus::PrimalForm0 g( calculus );
  double min_g = 300.0, max_g = 0.0;
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
    255.0;
      min_g = std::min( min_g , g.myContainer( index ) );
      max_g = std::max( max_g , g.myContainer( index ) );
    }
  trace.info() << "min_g= " << min_g << " max_g= " << max_g << std::endl;
  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = g;
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  Calculus::PrimalForm1 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1;

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 	primal_D0 = calculus.derivative<0,PRIMAL>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 	primal_D1 = calculus.derivative<1,PRIMAL>();
  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0       dual_D0   = calculus.derivative<0,DUAL>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1       dual_D1   = calculus.derivative<1,DUAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0          primal_h0 = calculus.hodge<0,PRIMAL>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1          primal_h1 = calculus.hodge<1,PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2          primal_h2 = calculus.hodge<2,PRIMAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1         	dual_h1   = calculus.hodge<1,DUAL>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2            dual_h2   = calculus.hodge<2,DUAL>();

  const DGtal::Dimension dimX = 0, dimY = 1;

  // BEG JOL
  // Calculus::PrimalIdentity1 tSS = calculus.identity<1, PRIMAL>();
  // for ( Calculus::Index index_i = 0; index_i < tSS.myContainer.rows(); index_i++ )
  //   for ( Calculus::Index index_j = 0; index_j < tSS.myContainer.cols(); index_j++ )
  //     {
  //       tSS.myContainer.coeffRef( index_i, index_j ) = 0.0;
  //       for ( Calculus::Index index_k = 0; index_k < tSS.myContainer.rows(); index_k++ )
  //           tSS.myContainer.coeffRef( index_i, index_j ) +=  sharp_x.myContainer.coeffRef( index_k, index_i ) * sharp_x.myContainer.coeffRef( index_k, index_j )
  //                                                          + sharp_y.myContainer.coeffRef( index_k, index_i ) * sharp_y.myContainer.coeffRef( index_k, index_j ) ;
  //     }
  // END JOL
  trace.endBlock();


  const Calculus::PrimalIdentity0 Id0 = calculus.identity<0, PRIMAL>();
  const Calculus::PrimalIdentity1 Id1 = calculus.identity<1, PRIMAL>();
  const Calculus::PrimalIdentity2 Id2 = calculus.identity<2, PRIMAL>();

  // Weight matrices
  //  Calculus::DualIdentity2   G0 		= ( 1.0/(h*h) ) * calculus.identity<2, DUAL>();
  Calculus::PrimalIdentity0 G0 	  = Id0; //	= ( 1.0/(h*h) ) * calculus.identity<0, PRIMAL>();
  Calculus::PrimalIdentity0 invG0 = Id0; //   = 	(h*h) 	* calculus.identity<0, PRIMAL>();

  //  Calculus::DualIdentity1   G1 		= calculus.identity<1, DUAL>();
  Calculus::PrimalIdentity1 G1    = Id1; //	= calculus.identity<1, PRIMAL>();
  Calculus::PrimalIdentity1 invG1 = Id1; //     = calculus.identity<1, PRIMAL>();

  //  Calculus::DualIdentity0   G2 		= 		(h*h) 	* calculus.identity<0, DUAL>();
  Calculus::PrimalIdentity2 G2    = Id2; //	= 		(h*h) 	* calculus.identity<2, PRIMAL>();
  Calculus::PrimalIdentity2 invG2 = Id2; //     = ( 1.0/(h*h) ) * calculus.identity<2, PRIMAL>();


  trace.beginBlock("precomputation of the structure tensor");

  constexpr typename DGtal::Dimension N = 2;

  ImageDouble imDouble = GenericReader<ImageDouble>::import( f1 );
  ImageSimpleMatrix2d T( domain );
  ImageDouble Ix = imDouble, Iy = imDouble;

  //structureTensor(T, imDouble, s, r);
  structureTensor(T, Ix, Iy, imDouble, s, r);

  // eigenvalues & eigenvectors
  ImageDouble vp1(domain), vp2(domain);
  ImageVector v1(domain), v2(domain);
  double vp1_max = 0.0;

  ImageDoubleIterator itvp1 = vp1.begin(), itvp2 = vp2.begin(); // ATTENTION rajouter les ++itvp1 et ++itvp2 dans la boucle ci dessous
  ImageVectorIterator itv1 = v1.begin(), itv2 = v2.begin();

  for ( ImageSimpleMatrix2dIterator itT = T.begin(); itT != T.end(); ++itv1, ++itv2, ++itvp1, ++itvp2, ++itT )
    {
      SimpleMatrix<double,2,2> Tij;
      Tij = *itT;

      SimpleMatrix<double,2,2> V;
      RealVector values;
      EigenDecomposition<2,double>::getEigenDecomposition( Tij, V, values );
      if (values[0] >= values[1])
        {
        *itvp1 = values[0];
        *itvp2 = values[1];
        (*itv1)[0] = V(0,0); (*itv1)[1] = V(1,0);
        (*itv2)[0] = V(0,1); (*itv2)[1] = V(1,1);
        }
      else
        {
        *itvp1 = values[1];
        *itvp2 = values[0];
        (*itv1)[0] = V(0,1); (*itv1)[1] = V(1,1);
        (*itv2)[0] = V(0,0); (*itv2)[1] = V(1,0);
        }

      vp1_max = max( vp1_max, *itvp1 );
    }
  trace.info() << "max_vp1=" << vp1_max << std::endl;

  if ( N == 2 )
    {
      DGtal::VTKWriter<Domain>( f2+"-vp1", vp1.domain() ) 	<< "data" << vp1;
      DGtal::VTKWriter<Domain>( f2+"-Ix", Ix.domain() ) << "data" << Ix;
      DGtal::VTKWriter<Domain>( f2+"-Iy", Iy.domain() ) << "data" << Iy;
    }

  // Computation of n=(nx,ny) at each e_i

  //Calculus::PrimalForm1 v_n = v;
  Calculus::PrimalForm1 n_edge = v;
  Calculus::PrimalForm1 n_edge_x = v, n_edge_y = v, norm2_edge = v;

  for ( Calculus::Index index = 0; index < v.myContainer.rows(); ++index)
    {
      // extraction des normales des 2 0-cellules adjacentes a l'arete consideree
      const Calculus::SCell& c = v.getSCell(index);
      const Calculus::Cell& uc = K.unsigns(c);
      Cells cells = K.uFaces(uc);
//      Dimension d = K.uOrthDir(uc);

      double eival1a = vp1(K.uCoords(cells[0]));
      double eival1b = vp1(K.uCoords(cells[1]));
      double eival2a = vp2(K.uCoords(cells[0]));
      double eival2b = vp2(K.uCoords(cells[1]));
      double mev1 = (eival1a + eival1b) / 2.0;
      double mev2 = (eival2a + eival2b) / 2.0;
      double mev1_sur_vp1_max = mev1 / vp1_max;
      //mev1_sur_vp1_max *= mev1_sur_vp1_max;

      double t1 = (2.0*mev2) / (mev1*s*s + e);
      double t2 = sqrt(mev1) / (0.9*vp1_max);

      RealVector n1 = v1(K.uCoords(cells[0]));
      RealVector n2 = v1(K.uCoords(cells[1]));
      double mnx = std::fabs(n1[0] + n2[0])/2.0;
      double mny = std::fabs(n1[1] + n2[1])/2.0;
      double norm2_mn = sqrt(mnx*mnx + mny*mny);
      mnx /= norm2_mn;
      mny /= norm2_mn;
//      n_edge_x.myContainer( index ) = mev1_sur_vp1_max*mnx + (1-mev1_sur_vp1_max); //mnx /= norm2_mn;
//      n_edge_y.myContainer( index ) = mev1_sur_vp1_max*mny + (1-mev1_sur_vp1_max); //mny /= norm2_mn;
      n_edge_x.myContainer( index ) = (1-t1)*t2*mnx + (1-(1-t1)*t2); //mnx /= norm2_mn;
      n_edge_y.myContainer( index ) = (1-t1)*t2*mny + (1-(1-t1)*t2); //mny /= norm2_mn;

      //double norm1_v1 = fabs(mnx) + fabs(mny);

//      if ( *(K.sDirs(c)) == 0 )
//        v.myContainer( index ) *= n_edge_x.myContainer( index );
//      else
//        v.myContainer( index ) *= n_edge_y.myContainer( index );

      norm2_edge.myContainer( index ) = sqrt(n_edge_x.myContainer( index )*n_edge_x.myContainer( index ) + n_edge_y.myContainer( index )*n_edge_y.myContainer( index ));
    }


  savePrimalForm1ToImage( calculus, dbl_image, n_edge_x, f2 + "-nx.pgm");
  savePrimalForm1ToImage( calculus, dbl_image, n_edge_y, f2 + "-ny.pgm");
  savePrimalForm1ToImage( calculus, dbl_image, norm2_edge, f2 + "-norm2.pgm");

  trace.endBlock();

  typedef Calculus::PrimalDerivative0::Container Matrix;


  // Canny edge detector
  Calculus::PrimalForm0 canny = g;
  Calculus::PrimalForm0 dI = g;
  double grad_conf = 5.0;
  double max_dI = 0.0;

  for ( Calculus::Index index = 0; index < canny.myContainer.rows(); ++index )
    {
      const Calculus::SCell& c = canny.getSCell(index);
      const int i = K.sCoord(c,0);
      const int j = K.sCoord(c,1);
//      const Point ij (i,j);

      RealVector nu = v1(K.sCoords(c));
      const double norm2_nu = std::sqrt( nu[0]*nu[0] + nu[1]*nu[1] );
      nu[0] /= norm2_nu;
      nu[1] /= norm2_nu;
      const double x = nu[0];
      const double y = nu[1];

      const double Ix_ij         = Ix(K.sCoords(c)); // Ix(i,j)
      const double Iy_ij         = Iy(K.sCoords(c)); // Iy(i,j)

      const double Ix_ij_plus_n  = interpolation(calculus, Ix, i, j,  x,  y);
      const double Iy_ij_plus_n  = interpolation(calculus, Iy, i, j,  x,  y);

      const double Ix_ij_minus_n = interpolation(calculus, Ix, i, j, -x, -y);
      const double Iy_ij_minus_n = interpolation(calculus, Iy, i, j, -x, -y);

//      std::cout << " Ix(i,j)= " << Ix_ij
//                << " Iy(i,j)= " << Iy_ij
//                << " Ix(i+x,j+x)= " << Ix_ij_plus_n
//                << " Iy(i+x,j+y)= " << Iy_ij_plus_n
//                << " Ix(i-x,j-x)= " << Ix_ij_minus_n
//                << " Iy(i-x,j-y)= " << Iy_ij_minus_n
//                << std::endl;


      // |G| = |Gx| + |Gy|
      const double grad_I           = std::fabs(Ix_ij)         + std::fabs(Iy_ij);
      const double grad_I_plus_n    = std::fabs(Ix_ij_plus_n)  + std::fabs(Iy_ij_plus_n);
      const double grad_I_minus_n   = std::fabs(Ix_ij_minus_n) + std::fabs(Iy_ij_minus_n);

      dI.myContainer( index ) = grad_I;
      max_dI = std::max( max_dI , grad_I );

//      std::cout << " (i,j)= (" << i << "," << j << ")"
//                << " (x,y)= (" << x << "," << y << ")"
//                << " grad_I(i,j)= "     << grad_I
//                << " grad_I(i+x,j+y)= " << grad_I_plus_n
//                << " grad_I(i-x,j-y)= " << grad_I_minus_n
//                << std::endl;

      if (grad_I > grad_I_plus_n && grad_I > grad_I_minus_n && grad_I > grad_conf)
          canny.myContainer( index ) = 0.0;
      else
          canny.myContainer( index ) = 1.0;
    }

  savePrimalForm0ToImage( calculus, end_image, canny, f2 + "-canny.pgm" );
  dI.myContainer /= max_dI;
  savePrimalForm0ToImage( calculus, end_image, dI, f2 + "-grad_I.pgm" );

  // Building diag(alpha)
  Calculus::PrimalIdentity0 diag_alpha = Id0;
  Calculus::PrimalForm0 alpha_var = g;
  double alpha_max = 0.0, alpha_min = 1.0;
  for ( Calculus::Index index = 0; index < canny.myContainer.rows(); ++index )
    {
      const Calculus::SCell& c = g.getSCell(index);
      const Calculus::Cell& uc = K.unsigns(c);
      const double one_minus_vp1_sur_vp1_max = std::max( 0.0 , 1.0 - (vp1(K.uCoords(uc)) / (0.9*vp1_max))*(vp1(K.uCoords(uc)) / (0.9*vp1_max)) );

      diag_alpha.myContainer.coeffRef( index, index )  = a * canny.myContainer( index );
      //diag_alpha.myContainer.coeffRef( index, index )  = a * one_minus_vp1_sur_vp1_max;

      alpha_var.myContainer( index ) = a * canny.myContainer( index );
      //alpha_var.myContainer( index ) = a * one_minus_vp1_sur_vp1_max;
      alpha_min = std::min( alpha_min , alpha_var.myContainer( index ) );
      alpha_max = std::max( alpha_max , alpha_var.myContainer( index ) );
    }

  alpha_var.myContainer /= alpha_max;
  savePrimalForm0ToImage( calculus, end_image, alpha_var, f2 + "-a_var.pgm" );
  trace.info() << "min_alpha=" << alpha_min << " max_alpha=" << alpha_max << std::endl;

  // Building alpha_G0_1
  const Calculus::PrimalIdentity0 alpha_iG0 = diag_alpha; //a * calculus.identity<0, PRIMAL>(); // a * invG0;
  const Calculus::PrimalForm0 alpha_iG0_g   = alpha_iG0 * g;

  // Building tA_A
  const Matrix& A = primal_D0.myContainer;
  const Matrix tA = A.transpose();
  Calculus::PrimalIdentity1 tA_A = G1;
  G1.myContainer = tA * A;

  // Building tS_S
  Calculus::PrimalAntiderivative1   sharp_x   = calculus.sharpDirectional<PRIMAL>(dimX);
  Calculus::PrimalAntiderivative1   sharp_y   = calculus.sharpDirectional<PRIMAL>(dimY);
  const Calculus::PrimalDerivative0 flat_x    = calculus.flatDirectional<PRIMAL>(dimX);
  const Calculus::PrimalDerivative0 flat_y    = calculus.flatDirectional<PRIMAL>(dimY);
  // All combinations below give the same result, which is strangely not an averaging.
  const Matrix& Sx = sharp_x.myContainer;
  const Matrix tSx = flat_x.myContainer; // Sx.transpose();
  const Matrix& Sy = sharp_y.myContainer;
  const Matrix tSy = flat_y.myContainer; // Sy.transpose();
  Calculus::PrimalIdentity1 tS_S = G1;
  tS_S.myContainer = (tSx * Sx + tSy * Sy); // (1.0/(h*h))*(tSx * Sx + tSy * Sy);
  // Calculus::PrimalIdentity1 tSx_Sx = G1;
  // tSx_Sx.myContainer = (tSx * Sx); // (1.0/(h*h))*(tSx * Sx + tSy * Sy);
  // Calculus::PrimalIdentity1 tSy_Sy = G1;
  // tSy_Sy.myContainer = (tSy * Sy); // (1.0/(h*h))*(tSx * Sx + tSy * Sy);

  // Building tA_tS_S_A
  Calculus::PrimalIdentity0 tA_tS_S_A = G0;
  tA_tS_S_A.myContainer = tA * tS_S.myContainer * A;

  // Builds a Laplacian but at distance 2 !
  // const Matrix& M = tA_tS_S_A.myContainer;
  // for (int k = 0; k < M.outerSize(); ++k)
  //   {
  //     trace.info() << "-----------------------------------------------------------------------" << endl;
  //     for ( Matrix::InnerIterator it( M, k ); it; ++it )
  //       trace.info() << "[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
  //   }


  // Building iG1_A_G0_tA_iG1 + tB_iG2_B
//  // A.tA
//  const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( invG1 * primal_D0 * G0 * dual_h2 * dual_D1 * primal_h1 * invG1 );
//  // tB.B
//  const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( dual_h1 * dual_D0 * primal_h2 * invG2 * primal_D1 );
  // tB.B + A.tA
  const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( invG1 * primal_D0 * G0 * dual_h2 * dual_D1 * primal_h1 * invG1
                                                            + dual_h1 * dual_D0 * primal_h2 * invG2 * primal_D1 );

  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;

  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "B'B'" << endl;
      const Calculus::PrimalIdentity1 lBB = l * lap_operator_v;
      Calculus::PrimalForm1 l_sur_4( calculus );
      for ( Calculus::Index index = 0; index < l_sur_4.myContainer.rows(); index++)
        l_sur_4.myContainer( index ) = l/4.0;
      l_sur_4 = Id1 * l_sur_4; //tS_S * l_sur_4; //
      double coef_eps = 1.0;
      double eps = coef_eps*e;

      for( int k = 0 ; k < 5 ; ++k )
        {
          if (eps/coef_eps < 2*h)
            break;
          else
            {
              eps /= coef_eps;
              Calculus::PrimalIdentity1 BB = eps * lBB + ( l/(4.0*eps) ) * Id1; // tS_S;
              int i = 0;
              for ( ; i < n; ++i )
                {
                  trace.info() << "------ Iteration " << k << ":" << 	i << "/" << n << " ------" << endl;
                  trace.beginBlock("Solving for u");
                  trace.info() << "Building matrix Av2A" << endl;

                  //double tvtSSv = 0.0;
                  Calculus::PrimalIdentity1 diag_v = diag( calculus, v );
                  Calculus::PrimalDerivative0 v_A = diag_v * primal_D0;
                  // Calculus::PrimalDerivative0 tS_S_v_A = tS_S * v_A;
                  // Calculus::PrimalIdentity0 Av2A = calculus.identity<0, PRIMAL>();
                  // Av2A.myContainer = v_A.myContainer.transpose() * tS_S_v_A.myContainer
                  //   + alpha_iG0.myContainer;
                  Calculus::PrimalIdentity0 Av2A = square( calculus, v_A ) + alpha_iG0;
                  trace.info() << "Prefactoring matrix Av2A := tv_tS_S_v.tA_tS_S_A + alpha_iG0" << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;
                  // const Matrix& M = Av2A.myContainer;
                  // for (int k = 0; k < M.outerSize(); ++k)
                  //   for ( Matrix::InnerIterator it( M, k ); it; ++it )
                  //     trace.info() << "[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
                  solver_u.compute( Av2A );
                  trace.info() << "Solving Av2A u = ag" << endl;
                  u = solver_u.solve( alpha_iG0_g );
                  trace.info() << ( solver_u.isValid() ? "OK" : "ERROR" ) << " " << solver_u.myLinearAlgebraSolver.info() << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;
                  trace.endBlock();

                  const Calculus::PrimalForm1 former_v = v;
                  trace.beginBlock("Solving for v");
                  trace.info() << "Building matrix BB+Mw2" << endl;
                  const Calculus::PrimalIdentity1 A_u = diag( calculus, primal_D0 * u );
                  const Calculus::PrimalIdentity1 tu_tA_A_u = square( calculus, A_u );
                  solver_v.compute( tu_tA_A_u + BB );
                  v = solver_v.solve( (1.0/eps) * l_sur_4 );
                  trace.info() << ( solver_v.isValid() ? "OK" : "ERROR" ) << " " << solver_v.myLinearAlgebraSolver.info() << endl;
                  trace.endBlock();

                  trace.beginBlock("Checking v, computing norms");
                  double m1 = 1.0;
                  double m2 = 0.0;
                  double ma = 0.0;
                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                    {
                      double val = v.myContainer( index );
                      m1 = std::min( m1, val );
                      m2 = std::max( m2, val );
                      ma += val;
                    }
                  trace.info() << "1-form v: min=" << m1 << " avg=" << ( ma/ v.myContainer.rows() )
                               << " max=" << m2 << std::endl;
                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                    v.myContainer( index ) = std::min( std::max(v.myContainer( index ), 0.0) , 1.0 );

                  double n_infty = 0.0;
                  double n_2 = 0.0;
                  double n_1 = 0.0;

                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                    {
                      n_infty = max( n_infty, fabs( v.myContainer( index ) - former_v.myContainer( index ) ) );
                      n_2    += ( v.myContainer( index ) - former_v.myContainer( index ) )
                                * ( v.myContainer( index ) - former_v.myContainer( index ) );
                      n_1    += fabs( v.myContainer( index ) - former_v.myContainer( index ) );
                    }
                  n_1 /= v.myContainer.rows();
                  n_2 = sqrt( n_2 / v.myContainer.rows() );

                  trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
                  trace.info() << "Variation |v^k+1 - v^k|_2 = " << n_2 << endl;
                  trace.info() << "Variation |v^k+1 - v^k|_1 = " << n_1 << endl;
                  trace.endBlock();
                  if ( n_infty < 1e-4 ) break;
                }
            }
        }

      // affichage des energies ********************************************************************

      trace.beginBlock("Computing energies");

      // Computation of <v,n>
      for ( Calculus::Index index = 0; index < v.myContainer.rows(); ++index)
        {
          const Calculus::SCell& c = v.getSCell(index);

//          if ( *(K.sDirs(c)) == 0 )
//            v.myContainer( index ) *= n_edge_x.myContainer( index );
//          else
//            v.myContainer( index ) *= n_edge_y.myContainer( index );
        }

      // a(u-g)^2
      const Calculus::PrimalForm0 u_minus_g = u - g;
      double alpha_square_u_minus_g = innerProduct( calculus, diag_alpha * u_minus_g, u_minus_g );
      trace.info() << "- a(u-g)^2      = " << alpha_square_u_minus_g << std::endl;
      // v^2|grad u|^2

      const Calculus::PrimalIdentity1 diag_v = diag( calculus, v );
      const Calculus::PrimalForm1 v_A_u = diag_v * primal_D0 * u;
      double square_v_grad_u = innerProduct( calculus, v_A_u, v_A_u );
      trace.info() << "- v^2|grad u|^2 = " << square_v_grad_u << std::endl;
//      // JOL: 1000 * plus rapide !
//      trace.info() << "  - u^t N u" << std::endl;
//      Calculus::PrimalForm0 u_prime = Av2A * u;
//      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
//        V2gradU2 += u.myContainer( index ) * u_prime.myContainer( index );
//      // for ( Calculus::Index index_i = 0; index_i < u.myContainer.rows(); index_i++)
//      // 	for ( Calculus::Index index_j = 0; index_j < u.myContainer.rows(); index_j++)
//      //     V2gradU2 += u.myContainer( index_i ) * Av2A.myContainer.coeff( index_i,index_j ) * u.myContainer( index_j ) ;

//      // le|grad v|^2
      Calculus::PrimalForm1 v_prime = lap_operator_v * v;
      double le_square_grad_v = l * eps * innerProduct( calculus, v, v_prime );
      trace.info() << "- le|grad v|^2  = " << le_square_grad_v << std::endl;

      // l(1-v)^2/4e
      Calculus::PrimalForm1 one_minus_v = v;
      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
        one_minus_v.myContainer( index_i ) = 1.0 - one_minus_v.myContainer( index_i );
      double l_over_4e_square_1_minus_v
        = l / (4*eps) * innerProduct( calculus, one_minus_v, one_minus_v );
      trace.info() << "- l(1-v)^2/4e   = " << l_over_4e_square_1_minus_v << std::endl;

      // l.per
      double Lper = 2.0* l_over_4e_square_1_minus_v; //le_square_grad_v + l_over_4e_square_1_minus_v;
      trace.info() << "- l.per         = " << Lper << std::endl;

      // AT tot
      double ATtot = alpha_square_u_minus_g + square_v_grad_u + Lper;

//      //      double per = 0.0;
//      //      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
//      //      {
//      //        per += (1/(4*e)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
//      //        for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
//      //            per += e * v.myContainer( index_i ) * tBB.myContainer( index_i,index_j ) * v.myContainer( index_j );
//      //      }


      // f << "l  " << "  a  " << "  e  " << "  a(u-g)^2  " << "  v^2|grad u|^2  " << "  le|grad v|^2  " << "  l(1-v)^2/4e  " << "  l.per  " << "  AT tot"<< endl;
      f << tronc(l,8) << "\t" << a << "\t"  << tronc(eps,4)
        << "\t" << tronc(alpha_square_u_minus_g,5)
        << "\t" << tronc(square_v_grad_u,5)
        << "\t" << tronc(le_square_grad_v,5)
        << "\t" << tronc(l_over_4e_square_1_minus_v,5)
        << "\t" << tronc(Lper,5)
        << "\t" << tronc(ATtot,5) << endl;

      trace.endBlock();

      // ***********************************************************************************************************************

      int int_l = (int) floor(l);
      int dec_l = (int) (floor((l-floor(l))*10000000));

      ostringstream ossU;
      ossU << boost::format("%s-l%.7f-u.pgm") %f2 %l;
      string str_image_u = ossU.str();
      savePrimalForm0ToImage( calculus, end_image, u, str_image_u);

      ostringstream ossV;
      ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
      string str_image_v = ossV.str();
      savePrimalForm1ToImage( calculus, dbl_image, v, str_image_v );
//      {
//        // Board2D board;
//        // board << calculus;
//        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
//        // ostringstream oss;
//        // oss << f2 << "-u-" << i << ".eps";
//        // string str_u = oss.str(); //f2 + "-u-" + .eps";
//        // board.saveEPS( str_u.c_str() );
//        Image end_image = image;
//        PrimalForm0ToImage( calculus, u, end_image );
//        ostringstream ossU;
//        //ossU << f2 << "-l" << int_l << "_" << dec_l << "-u.pgm";
//        ossU << boost::format("%s-l%.7f-u.pgm") %f2 %l;
//        string str_image_u = ossU.str();
//        end_image >> str_image_u.c_str();
//      }
//      {
//        // Board2D board;
//        // board << calculus;
//        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
//        //       << v;
//        // ostringstream oss;
//        // oss << f2 << "-v-" << i << ".eps";
//        // string str_v = oss.str();
//        // board.saveEPS( str_v.c_str() );
//        PrimalForm1ToImage( calculus, v, dbl_image );
//        ostringstream ossV;
//        //ossV << f2 << "-l" << int_l << "_" << dec_l << "-v.pgm";
//        ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
//        string str_image_v = ossV.str();
//        dbl_image >> str_image_v.c_str();
//      }
      l1 /= lr;
    }
  // typedef SelfAdjointEigenSolver<MatrixXd> EigenSolverMatrix;
  // const EigenSolverMatrix eigen_solver(laplace.myContainer);

  // const VectorXd eigen_values = eigen_solver.eigenvalues();
  // const MatrixXd eigen_vectors = eigen_solver.eigenvectors();

  // for (int kk=0; kk<laplace.myContainer.rows(); kk++)
  // {
  //     Calculus::Scalar eigen_value = eigen_values(kk, 0);

  //     const VectorXd eigen_vector = eigen_vectors.col(kk);
  //     const Calculus::DualForm0 eigen_form = Calculus::DualForm0(calculus, eigen_vector);
  //     std::stringstream ss;
  //     ss << "chladni_eigen_" << kk << ".svg";
  //     const std::string filename = ss.str();
  //     ss << "chladni_eigen_vector_" << kk << ".svg";
  //     trace.info() << kk << " " << eigen_value << " " << sqrt(eigen_value) << " " << eigen_vector.minCoeff() << " " << eigen_vector.maxCoeff() << " " << standard_deviation(eigen_vector) << endl;

  //     Board2D board;
  //     board << domain;
  //     board << calculus;
  //     board << CustomStyle("KForm", new KFormStyle2D(eigen_vectors.minCoeff(),eigen_vectors.maxCoeff()));
  //     board << eigen_form;
  //     board.saveSVG(filename.c_str());
  // }

  f.close();

  return 0;
}


