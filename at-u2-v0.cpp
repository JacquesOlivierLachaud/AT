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
void PrimalForm2ToImage( const Calculus& calculus, const typename Calculus::PrimalForm2& u, Image& image )
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
void savePrimalForm2ToImage( const Calculus& calculus, const Image& image, const typename Calculus::PrimalForm2& u, const string& filename )
{
    Image end_image = image;
    PrimalForm2ToImage( calculus, u, end_image );
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
    ("lambda-1", po::value<double>()->default_value( 0.3125 ), "the initial parameter lambda (l1)." ) // 0.3125
    ("lambda-2", po::value<double>()->default_value( 0.00005 ), "the final parameter lambda (l2)." )
    ("lambda-ratio,q", po::value<double>()->default_value( sqrt(2) ), "the division ratio for lambda from l1 to l2." )
    ("alpha,a", po::value<double>()->default_value( 1.0 ), "the parameter alpha." )
    ("epsilon,e", po::value<double>()->default_value( 1.0 ), "the initial and final parameter epsilon of AT functional at the same time." )
    ("epsilon-1", po::value<double>(), "the initial parameter epsilon." )
    ("epsilon-2", po::value<double>(), "the final parameter epsilon." )
    ("epsilon-r", po::value<double>()->default_value( 2.0 ), "sets the ratio between two consecutive epsilon values of AT functional." )
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
           << " / " << endl
           << " | a.(u-g)^2 + v^2 |grad u|^2 + le.|grad v|^2 + (l/4e).(1-v)^2 " << endl
           << " / " << endl
           << "Discretized as (u 2-form, v 0-form, A vertex-edge bdry, B edge-face bdy, M vertex-edge average)" << endl
           << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl
           << endl
           << general_opt << "\n"
           << "Example: ./at-u2-v0 -i ../Images/cerclesTriangle64b02.pgm -o tmp -a 0.05 -e 1 -1 0.1 -2 0.00001 -t 20"
           << endl;
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
  double e1 = vm.count( "epsilon-1" ) ? vm[ "epsilon-1" ].as<double>() : e;
  double e2 = vm.count( "epsilon-2" ) ? vm[ "epsilon-2" ].as<double>() : e;
  double er = vm[ "epsilon-r" ].as<double>();
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
  typedef Calculus::Index Index;
  Domain domain = image.domain();
  Point p0 = domain.lowerBound(); p0 *= 2;
  Point p1 = domain.upperBound(); p1 *= 2;
  p1      += Point::diagonal(1);
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
  Calculus::PrimalForm2 g( calculus );
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
        255.0;
    }
  trace.endBlock();

  // u = g at the beginning
  trace.info() << "u" << endl;
  Calculus::PrimalForm2 u = g;
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  Calculus::PrimalForm0 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1.0;
  Calculus::PrimalForm0 former_v = v;
  trace.info() << "v1 = 1" << endl;
  Calculus::PrimalForm1 v1( calculus );
  for ( Index index = 0; index < v1.myContainer.rows(); index++) 
    v1.myContainer( index ) = 1.0;

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 	primal_D0 = calculus.derivative<0,PRIMAL>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 	primal_D1 = calculus.derivative<1,PRIMAL>();
  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0       dual_D0   = calculus.derivative<0,DUAL>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1 	dual_D1   = calculus.derivative<1,DUAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0  	primal_h0 = calculus.hodge<0,PRIMAL>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1  	primal_h1 = calculus.hodge<1,PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2     	primal_h2 = calculus.hodge<2,PRIMAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1         	dual_h1   = calculus.hodge<1,DUAL>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2      	dual_h2   = calculus.hodge<2,DUAL>();

  // point_to_edge average operator
  Calculus::PrimalDerivative0   M01 = calculus.derivative<0, PRIMAL>();
  M01.myContainer = .5 * M01.myContainer.cwiseAbs();
  // edge_to_face average operator
  Calculus::PrimalDerivative1   M12 = calculus.derivative<1, PRIMAL>();
  M12.myContainer = .25 * M12.myContainer.cwiseAbs(); 
  const Calculus::PrimalAntiderivative2 primal_AD2 = dual_h1 * dual_D0 * primal_h2;

  trace.endBlock();

  const Calculus::PrimalIdentity0 Id0 = calculus.identity<0, PRIMAL>();
  const Calculus::PrimalIdentity1 Id1 = calculus.identity<1, PRIMAL>();
  const Calculus::PrimalIdentity2 Id2 = calculus.identity<2, PRIMAL>();

  typedef Calculus::PrimalDerivative0::Container Matrix;

  // Building alpha_G0_1
  const Calculus::PrimalIdentity2       alpha_Id2 = a * Id2;
  const Calculus::PrimalForm2           alpha_g   = a * g;
  const Calculus::PrimalIdentity0  lap_operator_v = -1.0 * dual_h2 * dual_D1 * primal_h1 * primal_D0;

  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 2, PRIMAL, 2, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverV;
  SolverV solver_v;

  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "B'B'" << endl;
      const Calculus::PrimalIdentity0 l_LAPV = l * lap_operator_v;
      Calculus::PrimalForm0 l_1_over_4( calculus );
      for ( Index index = 0; index < l_1_over_4.myContainer.rows(); index++)
        l_1_over_4.myContainer( index ) = l / 4.0;

      double eps = er*e;

      for ( double eps = e1; eps >= e2; eps /= er )
        {
//          if (eps/coef_eps < 2*h)
//            break;
//          else
//            {
//              eps /= coef_eps;
              Calculus::PrimalIdentity0 Per_Op      =  eps * l_LAPV + ( l/(4.0*eps) ) * Id0;
              Calculus::PrimalForm0     l_1_over_4e = (1.0/eps) * l_1_over_4;

              int i = 0;
              for ( ; i < n; ++i )
                {
                  trace.info() << "------ Iteration " << i << "/" << n << " ------" << endl;

                  trace.beginBlock("Solving for u as a 2-form");
                  trace.info() << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl;
                  trace.info() << "dE/du  = 2( a Id (u-g) + B diag(M v)^2 B^t u )" << endl;
                  trace.info() << "Building matrix U =  [ a Id + B diag(M v)^2 B^t ]" << endl;
                  Calculus::PrimalIdentity1 diag_v1 = diag( calculus, v1 );
                  Calculus::PrimalIdentity2 M_Id2   = -1.0 * primal_D1 * diag_v1 * diag_v1 * primal_AD2
                    + alpha_Id2;
                  trace.info() << "Prefactoring matrix U" << endl;
                  solver_u.compute( M_Id2 );
                  u = solver_u.solve( alpha_g );
                  trace.info() << "  => " << ( solver_u.isValid() ? "OK" : "ERROR" ) 
                               << " " << solver_u.myLinearAlgebraSolver.info() << endl;
                  trace.endBlock();
                  // E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)
                  // dE/dv  = 2( M^t diag( B^t u )^2 M v  + l e A^t A v  - l/4e Id (1-v) )
                  //  dE/dv = 0 <=> [ M^t diag( B^t u )^2 M + l e A^t A  + l/4e Id ] v = l/4e 1
                  trace.beginBlock("Solving for v");
                  former_v = v;
                  trace.info() << "E(u,v) = a(u-g)^t (u-g) +  u^t B diag(M v)^2 B^t u + l e v^t A^t A v + l/(4e) (1-v)^t (1-v)" << endl;
                  trace.info() << " 2( M^t diag( B^t u )^2 M v  + l e A^t A v  - l/4e Id (1-v) )" << endl;
                  trace.info() << "Building matrix V = [ M^t diag( B^t u )^2 M + l e A^t A  + l/4e Id ]" << endl;
                  Calculus::PrimalIdentity0 N_Id0 = Per_Op;
                  const Calculus::PrimalIdentity1 diag_Bt_u = diag( calculus, primal_AD2 * u );
                  N_Id0.myContainer += ( M01.transpose() * diag_Bt_u * diag_Bt_u * M01).myContainer;
                  trace.info() << "Prefactoring matrix V" << endl;
                  solver_v.compute( N_Id0 );
                  trace.info() << "Solving V v = l/4e * 1" << endl;
                  v = solver_v.solve( l_1_over_4e );
                  trace.info() << "  => " << ( solver_v.isValid() ? "OK" : "ERROR" ) 
                               << " " << solver_v.myLinearAlgebraSolver.info() << endl;
                  trace.info() << "Projecting v0 onto v1" << endl;
                  v1 = M01 * v;
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
                  if ( n_infty < 1e-4 ) break;
                }
//            }
        }
  
      // affichage des energies ********************************************************************

      trace.beginBlock("Computing energies");

       // a(u-g)^2
       const Calculus::PrimalForm2 u_minus_g = u - g;
       double alpha_square_u_minus_g = a * innerProduct( calculus, u_minus_g, u_minus_g );
       trace.info() << "- a(u-g)^2      = " << alpha_square_u_minus_g << std::endl;

       // v^2|grad u|^2
       const Calculus::PrimalIdentity1 diag_v1 = diag( calculus, v1 );
       const Calculus::PrimalForm1 v1_A_u = diag_v1 * primal_AD2 * u;
       double square_v1_grad_u = innerProduct( calculus, v1_A_u, v1_A_u );
       trace.info() << "- v^2|grad u|^2 = " << square_v1_grad_u << std::endl;
 //      // JOL: 1000 * plus rapide !
 //      trace.info() << "  - u^t N u" << std::endl;
 //      Calculus::PrimalForm0 u_prime = Av2A * u;
 //      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
 //        V2gradU2 += u.myContainer( index ) * u_prime.myContainer( index );
 //      // for ( Calculus::Index index_i = 0; index_i < u.myContainer.rows(); index_i++)
 //      // 	for ( Calculus::Index index_j = 0; index_j < u.myContainer.rows(); index_j++)
 //      //     V2gradU2 += u.myContainer( index_i ) * Av2A.myContainer.coeff( index_i,index_j ) * u.myContainer( index_j ) ;

 //      // le|grad v|^2
       Calculus::PrimalForm0 v_prime = lap_operator_v * v;
       double le_square_grad_v = l * eps * innerProduct( calculus, v, v_prime );
       trace.info() << "- le|grad v|^2  = " << le_square_grad_v << std::endl;

       // l(1-v)^2/4e
       Calculus::PrimalForm0 one_minus_v = v;
       for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
         one_minus_v.myContainer( index_i ) = 1.0 - one_minus_v.myContainer( index_i );
       double l_over_4e_square_1_minus_v
         = l / (4*eps) * innerProduct( calculus, one_minus_v, one_minus_v );
       trace.info() << "- l(1-v)^2/4e   = " << l_over_4e_square_1_minus_v << std::endl;

       // l.per
       double Lper = le_square_grad_v + l_over_4e_square_1_minus_v;
       trace.info() << "- l.per         = " << Lper << std::endl;

       // AT tot
       double ATtot = alpha_square_u_minus_g + square_v1_grad_u + Lper;

// //      //      double per = 0.0;
// //      //      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
// //      //      {
// //      //        per += (1/(4*e)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
// //      //        for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
// //      //            per += e * v.myContainer( index_i ) * tBB.myContainer( index_i,index_j ) * v.myContainer( index_j );
// //      //      }


       // f << "l  " << "  a  " << "  e  " << "  a(u-g)^2  " << "  v^2|grad u|^2  " << "  le|grad v|^2  " << "  l(1-v)^2/4e  " << "  l.per  " << "  AT tot"<< endl;
       f << tronc(l,8) << "\t" << a << "\t"  << tronc(eps,4)
         << "\t" << tronc(alpha_square_u_minus_g,5)
         << "\t" << tronc(square_v1_grad_u,5)
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
      savePrimalForm2ToImage( calculus, end_image, u, str_image_u);

      ostringstream ossV;
      ossV << boost::format("%s-l%.7f-v.pgm") %f2 %l;
      string str_image_v = ossV.str();
      savePrimalForm0ToImage( calculus, end_image, v, str_image_v);

      ostringstream ossV1;
      ossV1 << boost::format("%s-l%.7f-v1.pgm") %f2 %l;
      string str_image_v1 = ossV1.str();
      savePrimalForm1ToImage( calculus, dbl_image, v1, str_image_v1 );

//      {
//        // Board2D board;
//        // board << calculus;
//        // board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) ) << u;
//        // ostringstream oss;
//        // oss << f2 << "-u-" << i << ".eps";
//        // string str_u = oss.str(); //f2 + "-u-" + .eps";
//        // board.saveEPS( str_u.c_str() );
//        Image end_image = image;
//        PrimalForm2ToImage( calculus, u, end_image );
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
//        PrimalForm1ToImage( calculus, v1, dbl_image );
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


