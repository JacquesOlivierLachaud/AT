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
void DualForm0ToImage( const Calculus& calculus, const typename Calculus::DualForm0& u, Image& image )
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
void DualForm1ToImage( const Calculus& calculus, const typename Calculus::DualForm1& v, Image& image )
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
typename Calculus::DualIdentity1 diag( const Calculus& calculus,
                                       const typename Calculus::DualForm1& v )
{
  typename Calculus::DualIdentity1 diag_v = calculus.template identity<1, DUAL>();
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
typename Calculus::DualIdentity0 square( const Calculus& calculus,
                                         const typename Calculus::DualDerivative0& B )
{
  typename Calculus::DualIdentity0 tB_B = calculus.template identity<0, DUAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

//template <typename Calculus>
//typename Calculus::PrimalIdentity1 square( const Calculus& calculus,
//                                           const typename Calculus::DualIdentity1& B )
//{
//  typename Calculus::DualIdentity1 tB_B = calculus.template identity<1, DUAL>();
//  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;

//  typename Calculus::PrimalIdentity1 tB_B_primal = calculus.template identity<1, PRIMAL>();
//  tB_B_primal.myContainer = tB_B.myContainer;
//  return tB_B_primal;
//}

template <typename Calculus>
typename Calculus::DualIdentity1 square( const Calculus& calculus,
                                         const typename Calculus::DualIdentity1& B )
{
  typename Calculus::DualIdentity1 tB_B = calculus.template identity<1, DUAL>();
  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
  return tB_B;
}

//template <typename Calculus>
//typename Calculus::PrimalIdentity1 square( const Calculus& calculus,
//                                           const typename DGtal::LinearOperator<Calculus, 0, DUAL, 1, PRIMAL>& B )
//{
//  typename DGtal::LinearOperator<Calculus, 0, DUAL, 1, PRIMAL> tB_B = ?; //calculus.template identity<1, PRIMAL>();
//  tB_B.myContainer = B.myContainer.transpose() * B.myContainer;
//  return tB_B;
//}

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
template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::DualForm0& u,
                     const typename Calculus::DualForm0& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::DualForm1& u,
                     const typename Calculus::DualForm1& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
double innerProduct( const Calculus& calculus,
                     const typename Calculus::DualForm2& u,
                     const typename Calculus::DualForm2& v )
{
  double val = 0.0;
  for ( typename Calculus::Index index = 0; index < u.myContainer.rows(); index++ )
    val += u.myContainer( index ) * v.myContainer( index );
  return val;
}

template <typename Calculus>
void PrimalForm0ToDualForm0( const Calculus& calculus, typename Calculus::PrimalForm0& u, typename Calculus::DualForm0& u_star )
{
  const typename Calculus::PrimalHodge0 primal_h0 = calculus.template hodge<0,PRIMAL>();
  typename Calculus::DualForm2 u_2d = primal_h0 * u;
  typedef typename Calculus::KSpace KSpace;
  typedef typename KSpace::Cells Cells;
  typedef typename KSpace::Cell Cell;
  typedef typename KSpace::SCell SCell;
  const KSpace& K = calculus.myKSpace;
  for ( typename Calculus::Index index = 0; index < u_star.myContainer.rows(); index++)
    {
      const SCell& c = u_star.getSCell( index );
      const Cell& uc = K.unsigns(c);
      Cells cells = K.uFaces(uc);
      double u = 0.0;
      double n = 0.0;
      for ( typename Cells::ConstIterator it = cells.begin(),
          itE = cells.end(); it != itE; ++it )
      {
        Cell c = *it;
        if ( K.uDim( c ) == 0 )
        {
          u += u_2d.myContainer( calculus.template getCellIndex( c ) );
          n += 1.0;
        }
      }
      u_star.myContainer( index ) = u / n;
    }
}

template <typename Calculus>
void moy( const Calculus& calculus, const typename Calculus::PrimalForm0& g , typename Calculus::PrimalForm0& g_moy )
{
  //typename Calculus::PrimalForm0 g_moy( calculus );
  typedef typename Calculus::KSpace KSpace;
  typedef typename KSpace::Cells Cells;
  typedef typename KSpace::Cell Cell;
  typedef typename KSpace::SCell SCell;
  typedef typename KSpace::Point Point;
  typedef typename KSpace::Size Size;

  const KSpace& K = calculus.myKSpace;

  Size sKx  = K.size(0); // = w-1
  Size sKy  = K.size(1); // = h-1
  Point Klow ( 0.0 , 0.0 );
  Point Kup  ( 2*sKx , 2*sKy );

  for ( typename Calculus::Index index = 0; index < g_moy.myContainer.rows(); index++)
    {
      const SCell& c  = g_moy.getSCell( index );
      const Cell& uc  = K.unsigns(c);
      Cells coFaces_g = K.uCoFaces(uc);
      std::cout << "cell of g: " << c << std::endl;

      double s = g.myContainer( index );
      double n = 1.0;
      for ( typename Cells::ConstIterator it = coFaces_g.begin(),
          itE = coFaces_g.end(); it != itE; ++it )
      {
        Cell coFace = *it;
        Point pt = K.uKCoords(*it);
        std::cout << "  - " << pt;//(*it);
        if ( Klow.isLower(pt) && Kup.isUpper(pt) )
        {
            if ( K.uDim( coFace ) == 1 )
            {
                std::cout << "e ";
                std::cout << "[ " << Klow << ".isLower(pt) = " << Klow.isLower(pt) << ", " << Kup << ".isUpper(pt) = " << Kup.isUpper(pt) << "] [ ";
                Cells faces_of_edges_adjacent_to_g = K.uFaces(coFace);    // faces of an edge = 2 vertices

                //for ( typename Cells::ConstIterator it3 = faces_of_edges_adjacent_to_g.begin(),
                //    it3E = faces_of_edges_adjacent_to_g.end(); it3 != it3E; ++it3 )
                //      std::cout << (*it3) << "   " << std::endl;

                for ( typename Cells::ConstIterator it2 = faces_of_edges_adjacent_to_g.begin(),
                      it2E = faces_of_edges_adjacent_to_g.end(); it2 != it2E; ++it2 )
                {
                    Cell v = *it2;
                    std::cout << (*it2) << "   ";
                    //if ( calculus.template getCellIndex( v ) != calculus.template getCellIndex( c ) )
                    //if ( v != c )
                    if ( K.uDim( v ) == 0 )
                    {
                        s += g.myContainer( calculus.template getCellIndex( v ) );
                        n += 1.0;
                    }
                }
                std::cout << "]  ";
            }
        }
        std::cout << std::endl;
      }
      g_moy.myContainer( index ) = s / n;
      //std::cout << "  n = " << n << std::endl;
    }
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

  typedef std::vector< unsigned char >::iterator 						ImageIterator;
  typedef std::vector< double >::iterator 									ImageDoubleIterator;
  typedef std::vector< SimpleMatrix<double,2,2> >::iterator ImageSimpleMatrix2dIterator;
  typedef std::vector< RealVector >::iterator 							ImageVectorIterator;

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
  Point Klow (0.0,0.0);
  Point Kup  (t-2.0,t-2.0);
  calculus.myKSpace.init( Klow, Kup, true );
  const KSpace& K = calculus.myKSpace;

  // Les pixels sont des 0-cellules du primal.
  for ( Domain::ConstIterator it = kdomain.begin(), itE = kdomain.end(); it != itE; ++it )
    calculus.insertSCell( K.sCell( *it ) ); // ajoute toutes les cellules de Khalimsky.
  calculus.updateIndexes();
  trace.info() << calculus << endl;
  Calculus::PrimalForm0 g0( calculus );
  for ( Calculus::Index index = 0; index < g0.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g0.getSCell( index );
      g0.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
    255.0;
    }

  Calculus::DualForm0 g_star( calculus );
  PrimalForm0ToDualForm0( calculus , g0 , g_star );

  {
    Image im_g_star = image;
    DualForm0ToImage( calculus, g_star, im_g_star );
    ostringstream ossG_star;
    ossG_star << f2 << "-g_star.pgm";
    string str_image_g_star = ossG_star.str();
    im_g_star >> str_image_g_star.c_str();
  }

  Calculus::PrimalForm0 g( calculus );
  moy(calculus, g0, g);
  {
    Image end_image = image;
    PrimalForm0ToImage( calculus, g, end_image );
    ostringstream ossU;
    ossU << f2 << "-g.pgm";
    string str_image_u = ossU.str();
    end_image >> str_image_u.c_str();
  }

  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = g;
  trace.info() << "u*" << endl;
  Calculus::DualForm0 u_star( calculus );
  PrimalForm0ToDualForm0( calculus , u , u_star );
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  Calculus::PrimalForm1 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1;
  trace.info() << "v*" << endl;
  Calculus::DualForm1 v_star( calculus );
  for ( Calculus::Index index = 0; index < v_star.myContainer.rows(); index++)
    v_star.myContainer( index ) = 1;

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
  const Calculus::DualIdentity0  Id0_star = calculus.identity<0, DUAL>();
  const Calculus::DualIdentity1  Id1_star = calculus.identity<1, DUAL>();
  const Calculus::DualIdentity2  Id2_star = calculus.identity<2, DUAL>();

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

  Calculus::PrimalForm1 vG1( calculus );

  typedef Calculus::PrimalDerivative0::Container Matrix;

  // Building alpha_G0_1
  const Calculus::PrimalIdentity0 alpha_iG0  = a * calculus.identity<0, PRIMAL>(); // a * invG0;
  const Calculus::PrimalForm0 alpha_iG0_g    = alpha_iG0 * g;
  const Calculus::DualIdentity0 alpha_iG0_star  = a * calculus.identity<0, DUAL>(); // a * invG0;
  const Calculus::DualForm0 alpha_iG0_g_star = alpha_iG0_star * g_star;

  // Building tA_A
  const Matrix& A = primal_D0.myContainer;
  const Matrix tA = A.transpose();
  Calculus::PrimalIdentity1 tA_A = G1;
  //G1.myContainer = tA * A;
  tA_A.myContainer = tA * A;

  // Building tS_S
  const Calculus::PrimalAntiderivative1 sharp_x   = calculus.sharpDirectional<PRIMAL>(dimX);
  const Calculus::PrimalAntiderivative1 sharp_y   = calculus.sharpDirectional<PRIMAL>(dimY);
  const Calculus::PrimalDerivative0     flat_x    = calculus.flatDirectional<PRIMAL>(dimX);
  const Calculus::PrimalDerivative0     flat_y    = calculus.flatDirectional<PRIMAL>(dimY);
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

  // Building tS_S_star
  const Calculus::DualAntiderivative1   sharp_x_star   = calculus.sharpDirectional<DUAL>(dimX);
  const Calculus::DualAntiderivative1   sharp_y_star   = calculus.sharpDirectional<DUAL>(dimY);
  const Calculus::DualDerivative0       flat_x_star    = calculus.flatDirectional<DUAL>(dimX);
  const Calculus::DualDerivative0       flat_y_star    = calculus.flatDirectional<DUAL>(dimY);
  // All combinations below give the same result, which is strangely not an averaging.
  const Matrix& Sx_star = sharp_x_star.myContainer;
  const Matrix tSx_star = flat_x_star.myContainer; // Sx.transpose();
  const Matrix& Sy_star = sharp_y_star.myContainer;
  const Matrix tSy_star = flat_y_star.myContainer; // Sy.transpose();
  Calculus::DualIdentity1 tS_S_star = Id1_star;
  tS_S.myContainer = (tSx_star * Sx_star + tSy_star * Sy_star); // (1.0/(h*h))*(tSx * Sx + tSy * Sy);
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
  const Calculus::PrimalIdentity1 lap_operator_v = -1.0 * ( invG1 * primal_D0 * G0 * dual_h2 * dual_D1 * primal_h1 * invG1
                                                            + dual_h1 * dual_D0 * primal_h2 * invG2 * primal_D1 );
  const Calculus::DualIdentity1 lap_operator_v_star = -1.0 * ( primal_h1 * primal_D0 * dual_h2 * dual_D1
                                                            + dual_D0 * primal_h2 * primal_D1 * dual_h1);

  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, DUAL, 0, DUAL> SolverU_star;
  SolverU_star solver_u_star;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, DUAL, 1, DUAL> SolverV_star;
  SolverV_star solver_v_star;

  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "B'B'" << endl;
      const Calculus::PrimalIdentity1 lBB = l * lap_operator_v;
      const Calculus::DualIdentity1 lBB_star = l * lap_operator_v_star;
      Calculus::PrimalForm1 l_sur_4( calculus );
      for ( Calculus::Index index = 0; index < l_sur_4.myContainer.rows(); index++)
        l_sur_4.myContainer( index ) = l/4.0;
      Calculus::DualForm1 l_sur_4_star( calculus );
      l_sur_4_star.myContainer = l_sur_4.myContainer;
      l_sur_4 = Id1 * l_sur_4; //tS_S * l_sur_4; //
      l_sur_4_star = Id1_star * l_sur_4_star; //tS_S_star * l_sur_4_star; //
      double coef_eps = 2.0;
      double eps = coef_eps*e;

      for( int k = 0 ; k < 5 ; ++k )
        {
          if (eps/coef_eps < 2*h)
            break;
          else
            {
              eps /= coef_eps;
              Calculus::PrimalIdentity1 BB    = eps * lBB + ( l/(4.0*eps) ) * Id1; // tS_S;
              Calculus::DualIdentity1 BB_star = eps * lBB_star + ( l/(4.0*eps) ) * Id1_star; // tS_S;
              int i = 0;
              for ( ; i < n; ++i )
                {
                  trace.info() << "------ Iteration " << k << ":" << 	i << "/" << n << " ------" << endl;
                  trace.beginBlock("Solving for u");
                  trace.info() << "Building matrix Av2A" << endl;

                  //double tv_tS_S_v = 0.0;
                  Calculus::PrimalIdentity1 diag_v = diag( calculus, v );
                  Calculus::PrimalDerivative0 v_A = diag_v * primal_D0;
                  // Calculus::PrimalDerivative0 tS_S_v_A = tS_S * v_A;
                  // Calculus::PrimalIdentity0 Av2A = calculus.identity<0, PRIMAL>();
                  // Av2A.myContainer = v_A.myContainer.transpose() * tS_S_v_A.myContainer
                  //   + alpha_iG0.myContainer;
                  Calculus::PrimalIdentity0 Av2A = square( calculus, v_A ) + alpha_iG0;
                  trace.info() << "Prefactoring matrix Av2A := tA_diag(v*v)_A + alpha_iG0" << endl;
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

                  trace.beginBlock("Solving for u*");
                  trace.info() << "Building matrix Bv2B" << endl;

                  //double tv_tS_S_v = 0.0;
                  Calculus::DualIdentity1 diag_h1_v = diag( calculus, primal_h1 * v );
                  //DGtal::LinearOperator<Calculus, 0, DUAL, 1, PRIMAL> v_B = diag_v * dual_h1 * dual_D0;
                  Calculus::DualDerivative0 v_B = diag_h1_v * dual_D0;

                  // Calculus::PrimalDerivative0 tS_S_v_A = tS_S * v_A;
                  // Calculus::PrimalIdentity0 Av2A = calculus.identity<0, PRIMAL>();
                  // Av2A.myContainer = v_A.myContainer.transpose() * tS_S_v_A.myContainer
                  //   + alpha_iG0.myContainer;
                  Calculus::DualIdentity0 Bv2B = square( calculus, v_B ) + alpha_iG0_star;
                  trace.info() << "Prefactoring matrix Av2A := tA_diag(v*v)_A + alpha_iG0" << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;
//                  // const Matrix& M = Av2A.myContainer;
//                  // for (int k = 0; k < M.outerSize(); ++k)
//                  //   for ( Matrix::InnerIterator it( M, k ); it; ++it )
//                  //     trace.info() << "[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
                  solver_u_star.compute( Bv2B );
                  trace.info() << "Solving Bv2B u* = ag*" << endl;
                  u_star = solver_u_star.solve( alpha_iG0_g_star );
                  trace.info() << ( solver_u_star.isValid() ? "OK" : "ERROR" ) << " " << solver_u_star.myLinearAlgebraSolver.info() << endl;
                  trace.info() << "-------------------------------------------------------------------------------" << endl;

//                  PrimalForm0ToDualForm0( calculus , u , u_star );

                  trace.endBlock();

                  const Calculus::PrimalForm1 former_v = v;
                  trace.beginBlock("Solving for v");
                  trace.info() << "Building matrix BB+Mw2" << endl;
                  const Calculus::PrimalIdentity1 A_u = diag( calculus, primal_D0 * u );
                  const Calculus::PrimalIdentity1 tu_tA_A_u = square( calculus, A_u );
                  Calculus::PrimalForm1 one_minus_v_star = v;
                  for ( Calculus::Index index_i = 0; index_i < one_minus_v_star.myContainer.rows(); index_i++)
                    one_minus_v_star.myContainer( index_i ) = 1.0 - v_star.myContainer( index_i );
                  solver_v.compute( tu_tA_A_u + BB );
                  v = solver_v.solve( (1.0/eps) * l_sur_4 );
                  //v = solver_v.solve( (l/(4*eps)) * one_minus_v_star );
                  trace.info() << ( solver_v.isValid() ? "OK" : "ERROR" ) << " " << solver_v.myLinearAlgebraSolver.info() << endl;
                  trace.endBlock();

                  trace.beginBlock("Solving for v*");
                  trace.info() << "Building matrix BB+Mw2" << endl;
                  const Calculus::DualIdentity1 B_u_star = diag( calculus, dual_D0 * u_star );
                  const Calculus::DualIdentity1 tu_tB_B_u_star = square( calculus, B_u_star );
                  Calculus::DualForm1 one_minus_v = v_star;
                  for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
                    one_minus_v.myContainer( index_i ) = 1.0 - v.myContainer( index_i );
                  solver_v_star.compute( tu_tB_B_u_star + BB_star );
                  v_star = solver_v_star.solve( (1.0/eps) * l_sur_4_star );
                  //v_star = solver_v_star.solve( (l/(4*eps)) * one_minus_v );
                  trace.info() << ( solver_v_star.isValid() ? "OK" : "ERROR" ) << " " << solver_v_star.myLinearAlgebraSolver.info() << endl;
                  trace.endBlock();

//                  trace.beginBlock("Constraint v+v* = 1");
//                  Calculus::PrimalForm1 v_plus_v_star = v;
//                  v_plus_v_star.myContainer += v_star.myContainer;
//                  double m = 0.0;
//                  double M = 0.0;
//                  for ( Calculus::Index index = 0; index < v_plus_v_star.myContainer.rows(); index++)
//                  {
//                      m = std::min( m, v_plus_v_star.myContainer( index ) );
//                      M = std::max( M, v_plus_v_star.myContainer( index ) );
//                  }
//                  trace.info() << "1-form v+v*: min= " << m << " max=" << M << std::endl;

//                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
//                  {
//                      v.myContainer( index ) /= M;
//                      v_star.myContainer( index ) /= M;
//                  }
//                  trace.endBlock();

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
                      n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
                      n_2    += ( v.myContainer( index ) - former_v.myContainer( index ) )
                                * ( v.myContainer( index ) - former_v.myContainer( index ) );
                      n_1    += abs( v.myContainer( index ) - former_v.myContainer( index ) );
                    }
                  n_1 /= v.myContainer.rows();
                  n_2 = sqrt( n_2 / v.myContainer.rows() );

                  trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
                  trace.info() << "Variation |v^k+1 - v^k|_2 = " << n_2 << endl;
                  trace.info() << "Variation |v^k+1 - v^k|_1 = " << n_1 << endl;
                  trace.endBlock();

                  trace.beginBlock("Checking v*, computing norms");
                  double m1_star = 1.0;
                  double m2_star = 0.0;
                  double ma_star = 0.0;
                  for ( Calculus::Index index = 0; index < v_star.myContainer.rows(); index++)
                    {
                      double val = v_star.myContainer( index );
                      m1_star = std::min( m1_star, val );
                      m2_star = std::max( m2_star, val );
                      ma_star += val;
                    }
                  trace.info() << "1-form v*: min=" << m1_star << " avg=" << ( ma_star/ v_star.myContainer.rows() )
                               << " max=" << m2_star << std::endl;
                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
                      v_star.myContainer( index ) = std::min( std::max(v_star.myContainer( index ), 0.0) , 1.0 );


//                  double n_infty = 0.0;
//                  double n_2 = 0.0;
//                  double n_1 = 0.0;

//                  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
//                    {
//                      n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
//                      n_2    += ( v.myContainer( index ) - former_v.myContainer( index ) )
//                                * ( v.myContainer( index ) - former_v.myContainer( index ) );
//                      n_1    += abs( v.myContainer( index ) - former_v.myContainer( index ) );
//                    }
//                  n_1 /= v.myContainer.rows();
//                  n_2 = sqrt( n_2 / v.myContainer.rows() );

//                  trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
//                  trace.info() << "Variation |v^k+1 - v^k|_2 = " << n_2 << endl;
//                  trace.info() << "Variation |v^k+1 - v^k|_1 = " << n_1 << endl;
                  trace.endBlock();

                  if ( n_infty < 1e-4 ) break;
                }
            }
        }

      // affichage des energies ********************************************************************

      trace.beginBlock("Computing energies");

      // a(u-g)^2
      const Calculus::PrimalForm0 u_minus_g = u - g;
      double alpha_square_u_minus_g = a * innerProduct( calculus, u_minus_g, u_minus_g );
      trace.info() << "- a(u-g)^2      = " << alpha_square_u_minus_g << std::endl;

      // a(u*-g*)^2
      const Calculus::DualForm0 u_minus_g_star = u_star - g_star;
      double alpha_square_u_minus_g_star = a * innerProduct( calculus, u_minus_g_star, u_minus_g_star );
      trace.info() << "- a(u*-g*)^2    = " << alpha_square_u_minus_g_star << std::endl;

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

      // v*^2|grad u*|^2

      const Calculus::DualIdentity1 diag_v_star = diag( calculus, v_star );
      const Calculus::DualForm1 v_B_u = diag_v_star * dual_D0 * u_star;
      double square_v_grad_u_star = innerProduct( calculus, v_B_u, v_B_u );
      trace.info() << "- v*^2|grad u*|^2 = " << square_v_grad_u_star << std::endl;
//      // JOL: 1000 * plus rapide !
//      trace.info() << "  - u^t N u" << std::endl;
//      Calculus::PrimalForm0 u_prime = Av2A * u;
//      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
//        V2gradU2 += u.myContainer( index ) * u_prime.myContainer( index );
//      // for ( Calculus::Index index_i = 0; index_i < u.myContainer.rows(); index_i++)
//      // 	for ( Calculus::Index index_j = 0; index_j < u.myContainer.rows(); index_j++)
//      //     V2gradU2 += u.myContainer( index_i ) * Av2A.myContainer.coeff( index_i,index_j ) * u.myContainer( index_j ) ;

//      // le|grad v*|^2
      Calculus::DualForm1 v_prime_star = lap_operator_v_star * v_star;
      double le_square_grad_v_star = l * eps * innerProduct( calculus, v_star, v_prime_star );
      trace.info() << "- le|grad v*|^2  = " << le_square_grad_v_star << std::endl;

      // l(1-v)^2/4e
      Calculus::PrimalForm1 one_minus_v = v;
      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
        one_minus_v.myContainer( index_i ) = 1.0 - one_minus_v.myContainer( index_i );
      double l_over_4e_square_1_minus_v
        = l / (4*eps) * innerProduct( calculus, one_minus_v, one_minus_v );
      trace.info() << "- l(1-v)^2/4e   = " << l_over_4e_square_1_minus_v << std::endl;

      // l(1-v_v*)^2/4e
      Calculus::PrimalForm1 one_minus_v_minus_v_star = v;
      one_minus_v_minus_v_star.myContainer += v_star.myContainer;
      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
        one_minus_v_minus_v_star.myContainer( index_i ) = 1.0 - one_minus_v.myContainer( index_i );
      double l_over_4e_square_1_minus_v_minus_v_star
        = l / (4*eps) * innerProduct( calculus, one_minus_v_minus_v_star, one_minus_v_minus_v_star );
      trace.info() << "- l(1-v_v*)^2/4e   = " << l_over_4e_square_1_minus_v_minus_v_star << std::endl;

      // l.per
      //double Lper = le_square_grad_v + le_square_grad_v_star + l_over_4e_square_1_minus_v_minus_v_star;
      double Lper = le_square_grad_v + l_over_4e_square_1_minus_v;
      trace.info() << "- l.per         = " << Lper << std::endl;

      // AT tot
      double ATtot = alpha_square_u_minus_g + square_v_grad_u + Lper;
      //double ATtot = alpha_square_u_minus_g      + square_v_grad_u      + le_square_grad_v      + l_over_4e_square_1_minus_v_minus_v_star
      //             + alpha_square_u_minus_g_star + square_v_grad_u_star + le_square_grad_v_star ;

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

      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
        // ostringstream oss;
        // oss << f2 << "-u-" << i << ".eps";
        // string str_u = oss.str(); //f2 + "-u-" + .eps";
        // board.saveEPS( str_u.c_str() );
        Image end_image = image;
        PrimalForm0ToImage( calculus, u, end_image );
        ostringstream ossU;
        //ossU << f2 << "-l" << int_l << "_" << dec_l << "-u.pgm";
        ossU << boost::format("%s-l%.7f-u.pgm") %f2 %l;
        string str_image_u = ossU.str();
        end_image >> str_image_u.c_str();
      }
      {
          // Board2D board;
          // board << calculus;
          // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
          // ostringstream oss;
          // oss << f2 << "-u-" << i << ".eps";
          // string str_u = oss.str(); //f2 + "-u-" + .eps";
          // board.saveEPS( str_u.c_str() );
          Image end_image = image;
          DualForm0ToImage( calculus, u_star, end_image );
          ostringstream ossU;
          //ossU << f2 << "-l" << int_l << "_" << dec_l << "-u.pgm";
          ossU << boost::format("%s-l%.7f-u_star.pgm") %f2 %l;
          string str_image_u = ossU.str();
          end_image >> str_image_u.c_str();
        }
      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
        //       << v;
        // ostringstream oss;
        // oss << f2 << "-v-" << i << ".eps";
        // string str_v = oss.str();
        // board.saveEPS( str_v.c_str() );
        PrimalForm1ToImage( calculus, v, dbl_image );
        ostringstream ossV;
        //ossV << f2 << "-l" << int_l << "_" << dec_l << "-v.pgm";
        ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
        string str_image_v = ossV.str();
        dbl_image >> str_image_v.c_str();
      }
      {
        // Board2D board;
        // board << calculus;
        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
        //       << v;
        // ostringstream oss;
        // oss << f2 << "-v-" << i << ".eps";
        // string str_v = oss.str();
        // board.saveEPS( str_v.c_str() );
        DualForm1ToImage( calculus, v_star, dbl_image );
        ostringstream ossV;
        //ossV << f2 << "-l" << int_l << "_" << dec_l << "-v.pgm";
        ossV << boost::format("%s-l%.7f-v_star.pgm") %f2 %l;
        string str_image_v = ossV.str();
        dbl_image >> str_image_v.c_str();
      }
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


