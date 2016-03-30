

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
#include "RealFFT.h" 
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

double tronc( const double& nb, const int& p )
{
  int i = pow(10,p) * nb;
  return i/pow(10,p);
}

//void exportPrimalForm1( const Calculus& calculus, const typename Calculus::PrimalForm1& v, Image& image, const )
//{

//        PrimalForm1ToImage( calculus, v, image );
//        ostringstream oss;
//        //oss3 << f2 << "-l" << tronc_l << "-v.pgm";
//        oss3 << boost::format("%s-g.pgm") %f2;
//        string str_image_v = oss3.str();
//        
//        dbl_image >> str_image_v.c_str();


//}

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
  const KSpace& K = calculus.myKSpace;
  // Les pixels sont des 0-cellules du primal.
  for ( Domain::ConstIterator it = kdomain.begin(), itE = kdomain.end(); it != itE; ++it )
    calculus.insertSCell( K.sCell( *it ) ); // ajoute toutes les cellules de Khalimsky.
  calculus.updateIndexes();
  trace.info() << calculus << endl;
  Calculus::PrimalForm0 g( calculus );
  for ( Calculus::Index index = 0; index < g.myContainer.rows(); index++)
    {
      const Calculus::SCell& cell = g.getSCell( index );
      g.myContainer( index ) = ((double) image( K.sCoords( cell ) )) /
	255.0;
    }
  //  {
  //    Board2D board;
  //    board << calculus;
  //    board << CustomStyle( "KForm", new KFormStyle2D( 0.0, 1.0 ) )
  //          << g;
  //    string str_calculus = f2 + "-calculus.eps";
  //    board.saveEPS( str_calculus.c_str() );
  //  }
  trace.endBlock();
 
  trace.info() << "ag" << endl;
  const Calculus::PrimalForm0 ag = a * g;
  trace.info() << "u" << endl;
  Calculus::PrimalForm0 u = ag;
  // trace.info() << "A^t*diag(v)^2*A = " << Av2A << endl;	
  trace.info() << "v" << endl;
  Calculus::PrimalForm1 v( calculus );
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    v.myContainer( index ) = 1.0;
      

  trace.beginBlock("building AT functionnals");
  trace.info() << "primal_D0" << endl;
  const Calculus::PrimalDerivative0 	primal_D0 = calculus.derivative<0,PRIMAL>();
  trace.info() << "primal_h0" << endl;
  const Calculus::PrimalHodge0  			primal_h0 = calculus.hodge<0,PRIMAL>();
  trace.info() << "primal_h1" << endl;
  const Calculus::PrimalHodge1  			primal_h1 = calculus.hodge<1,PRIMAL>();
  trace.info() << "dual_D1" << endl;
  const Calculus::DualDerivative1 		dual_D1   = calculus.derivative<1,DUAL>();
  trace.info() << "dual_h2" << endl;
  const Calculus::DualHodge2      		dual_h2   = calculus.hodge<2,DUAL>();
  trace.info() << "primal_D1" << endl;
  const Calculus::PrimalDerivative1 	primal_D1 = calculus.derivative<1,PRIMAL>();
  trace.info() << "primal_h2" << endl;
  const Calculus::PrimalHodge2     		primal_h2 = calculus.hodge<2,PRIMAL>();
  trace.info() << "dual_D0" << endl;
  const Calculus::DualDerivative0     dual_D0   = calculus.derivative<0,DUAL>();
  trace.info() << "dual_h1" << endl;
  const Calculus::DualHodge1         	dual_h1   = calculus.hodge<1,DUAL>();
  trace.endBlock();
  
  
  trace.beginBlock("precomputation of the structure tensor");
    
  constexpr typename DGtal::Dimension N = 2;
 	
  ImageDouble imDouble = GenericReader<ImageDouble>::import( f1 );
  ImageSimpleMatrix2d T( domain );
  ImageDouble Ix( domain ), Iy( domain );

  structureTensor(T, Ix, Iy, imDouble, s, r);
  
  // eigenvalues & eigenvectors
  ImageDouble vp1(domain), vp2(domain);
  ImageVector v1(domain), v2(domain);
  double smax = 0.0;
   
  ImageDoubleIterator itvp1 = vp1.begin();//, itvp2 = vp2.begin(); ATTENTION rajouter les ++itvp1 et ++itvp2 dans la boucle ci dessous
  ImageVectorIterator itv1 = v1.begin(), itv2 = v2.begin();
  
  for ( ImageSimpleMatrix2dIterator itT = T.begin(); itT != T.end(); ++itv1, ++itv2, ++itvp1, ++itT )
    {
      SimpleMatrix<double,2,2> Tij;
      Tij = *itT;
      
      SimpleMatrix<double,2,2> V;
      RealVector values;
      EigenDecomposition<2,double>::getEigenDecomposition( Tij, V, values );
      if (values[0] >= values[1])
	{
	  *itvp1 = values[0];
	  //*itvp2 = values[1];	
	  (*itv1)[0] = V(0,0); (*itv1)[1] = V(1,0); 
	  (*itv2)[0] = V(0,1); (*itv2)[1] = V(1,1); 
	}
      else
	{
	  *itvp1 = values[1];
	  //*itvp2 = values[0];	
	  (*itv1)[0] = V(0,1); (*itv1)[1] = V(1,1); 
	  (*itv2)[0] = V(0,0); (*itv2)[1] = V(1,0); 
	}
      
      smax = max( smax, *itvp1 );
    }
  
  DGtal::VTKWriter<Domain>( f2+"-vp1", vp1.domain() ) 	<< "data" << vp1;
  
  
  
  // Weight matrix G1
  
  Calculus::PrimalIdentity1 G1 		= calculus.identity<1, PRIMAL>();
  Calculus::PrimalIdentity1 invG1 = calculus.identity<1, PRIMAL>();
  Calculus::PrimalForm1 vG1( calculus );
  
  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
    {	
      // extraction des normales des 2 0-cellules adjacentes a l'arete consideree
      const Calculus::SCell& c = v.getSCell(index);
      const Calculus::Cell& uc = K.unsigns(c);
      Cells cells = K.uFaces(uc);
      Dimension d = K.uOrthDir(uc);
      RealVector n1 = v1(K.uCoords(cells[0]));
      RealVector n2 = v1(K.uCoords(cells[1]));
      double eival1a = vp1(K.uCoords(cells[0]));
      double eival1b = vp1(K.uCoords(cells[1]));
      double eival2a = vp2(K.uCoords(cells[0]));
      double eival2b = vp2(K.uCoords(cells[1]));
      double mnx = (n1[0] + n2[0])/2.0;
      double mny = (n1[1] + n2[1])/2.0;
      double norm_mn = sqrt(mnx*mnx + mny*mny);
      mnx /= norm_mn;
      mny /= norm_mn;
      double meival1 = (eival1a + eival1b)/2.0;
      double meival2 = (eival2a + eival2b)/2.0; 

      //		double mnx = (imageIx(K.uCoords(s[0]))+imageIx(K.uCoords(s[1])))/2.0;
      //		double mny = (imageIy(K.uCoords(s[0]))+imageIy(K.uCoords(s[1])))/2.0; 

      double norm1_v1 = abs(mnx) + abs(mny);
      meival1 = min( smax, meival1 );
      meival2 = min( smax, meival2);
      double wG = 1.0 - pow(meival1/smax,2.0 ) * (1.0-1.0/norm1_v1);
      //double wG = d==0 ? abs(mny) : abs(mnx);
      //wG = std::max(wG,0.01);
			
      G1.myContainer.coeffRef( index, index )    = wG;
      invG1.myContainer.coeffRef( index, index ) = 1.0 / wG;
      vG1.myContainer( index ) = wG;
      //std::cout << G1.myContainer.coeffRef( index, index ) << ' ' ;
    }		
	
  {
    PrimalForm1ToImage( calculus, vG1, dbl_image );
    ostringstream ossG;
    ossG << f2 << "-G.pgm";
    string str_image_G = ossG.str();
    dbl_image >> str_image_G.c_str();
  }
  trace.endBlock();
  
  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 0, PRIMAL, 0, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;
  
  SolverV solver_lap_operator_v;
  const Calculus::PrimalIdentity1 lap_operator_v = -( 1.0 / h ) * ( primal_D0 * dual_h2 * dual_D1 * primal_h1 * invG1 
							    																								+ G1 * dual_h1 * dual_D0 * primal_h2 * primal_D1 );
  solver_lap_operator_v.compute( lap_operator_v );

  while ( l1 >= l2 )
    {
      trace.info() << "************ lambda = " << l1 << " **************" << endl;
      double l = l1;
      trace.info() << "B'B'" << endl;
      const Calculus::PrimalIdentity1 lBB = l * lap_operator_v;
      Calculus::PrimalForm1 l_4( calculus );
      for ( Calculus::Index index = 0; index < l_4.myContainer.rows(); index++)
        l_4.myContainer( index ) = l/4.0;
      
      Calculus::PrimalIdentity1 BB = 0.0 * lBB;
      
      double coef_eps = 2.0;
      double eps = coef_eps*e;
      
      for( int k = 0 ; k < 5 ; ++k )
	{  
	  if (eps/coef_eps < h*h)
	    break;
	  else
	    {
	      eps /= coef_eps;
	      BB = eps * lBB + (h*l/(4.0*eps)) * invG1;//calculus.identity<1, PRIMAL>();  
	      int i = 0;
	      for ( ; i < n; ++i )
		{
		  trace.info() << "------ Iteration " << k << ":" << 	i << "/" << n << " ------" << endl;
		  trace.beginBlock("Solving for u");
		  trace.info() << "Building matrix Av2A" << endl;
				      
		  Calculus::PrimalIdentity1 Mv2 = calculus.identity<1, PRIMAL>();
		  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
		    Mv2.myContainer.coeffRef( index, index ) = v.myContainer[ index ] * v.myContainer[ index ];
				      
		  const Calculus::PrimalIdentity0 Av2A = 	(- 1.0 / h) 	* dual_h2 * dual_D1 * primal_h1 * Mv2 * invG1 * primal_D0
		    																				+ (a * h*h) * calculus.identity<0, PRIMAL>();
		  trace.info() << "Prefactoring matrix Av2A" << endl;
		  solver_u.compute( Av2A );
		  trace.info() << "Solving Av2A u = ag" << endl;
		  u = solver_u.solve( (h*h) * ag );
		  trace.info() << ( solver_u.isValid() ? "OK" : "ERROR" ) << " " << solver_u.myLinearAlgebraSolver.info() << endl;
		  trace.endBlock();

		  trace.beginBlock("Solving for v");
		  trace.info() << "Building matrix BB+Mw2" << endl;
		  const Calculus::PrimalForm1 former_v = v;
		  const Calculus::PrimalForm1 w = primal_D0 * u;
				      
		  Calculus::PrimalIdentity1 Mw2 = calculus.identity<1, PRIMAL>();
		  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
		    Mw2.myContainer.coeffRef( index, index ) = (1.0/h) * w.myContainer[ index ] * w.myContainer[ index ];
				        
		  trace.info() << "Prefactoring matrix BB+Mw2" << endl;
		  solver_v.compute( BB + Mw2 * invG1 );
		  trace.info() << 	"Solving (BB+Mw2)v = l_4e" << endl;
		  v = solver_v.solve( (h * (1.0/eps)) * invG1 * l_4 );
		  trace.info() << ( solver_v.isValid() ? "OK" : "ERROR" ) << " " << solver_v.myLinearAlgebraSolver.info() << endl;
		  trace.endBlock();
				      
		  double n_infty = 0.0; 
		  for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
		    n_infty = max( n_infty, abs( v.myContainer( index ) - former_v.myContainer( index ) ) );
		  trace.info() << "Variation |v^k+1 - v^k|_oo = " << n_infty << endl;
		  if ( n_infty < 1e-4 ) break;
		}
	    }
	}
        
        
      // affichage des energies ***********************************************************************************
      
      typedef Calculus::SparseMatrix SparseMatrix;
      typedef Eigen::Matrix<double,Dynamic,Dynamic> Matrix;

      trace.beginBlock("Computing energies");

      // a(u-g)^2 
      trace.info() << "- a(u-g)^2 " << std::endl;
      double UmG2 = 0.0;
      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
        UmG2 += (a*h*h) * (u.myContainer( index ) - g.myContainer( index )) * (u.myContainer( index ) - g.myContainer( index ));
      
      // v^2|grad u|^2 
      trace.info() << "- v^2|grad u|^2" << std::endl;
      double V2gradU2 = 0.0;
      SolverU solver_Av2A;
      trace.info() << "  - Id" << std::endl;
      Calculus::PrimalIdentity1 Mv2 = calculus.identity<1, PRIMAL>();
      trace.info() << "  - M := Diag(v^2)" << std::endl;
      for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
        Mv2.myContainer.coeffRef( index, index ) = v.myContainer[ index ] * v.myContainer[ index ];
      trace.info() << "  - * D_1 * M * D_0" << std::endl;
      const Calculus::PrimalIdentity0 Av2A = - (1.0/h) * dual_h2 * dual_D1 * primal_h1 * Mv2 * invG1 * primal_D0;
      trace.info() << "  - N := compute (* D_1 * M * D_0)" << std::endl;
      solver_Av2A.compute( Av2A );
      // JOL: 1000 * plus rapide !
      trace.info() << "  - u^t N u" << std::endl;
      Calculus::PrimalForm0 u_prime = Av2A * u;
      for ( Calculus::Index index = 0; index < u.myContainer.rows(); index++)
        V2gradU2 += u.myContainer( index ) * u_prime.myContainer( index );
      // for ( Calculus::Index index_i = 0; index_i < u.myContainer.rows(); index_i++)
      // 	for ( Calculus::Index index_j = 0; index_j < u.myContainer.rows(); index_j++)
      //     V2gradU2 += u.myContainer( index_i ) * Av2A.myContainer.coeff( index_i,index_j ) * u.myContainer( index_j ) ;
      
      // le|grad v|^2 
      trace.info() << "- le|grad v|^2" << std::endl;
      Calculus::PrimalForm1 v_prime = lap_operator_v * v;
      double gradV2 = 0.0;
      for ( Calculus::Index index = 0; index < v.myContainer.rows(); index++)
        gradV2 += (l * eps / h) * v.myContainer( index ) * v_prime.myContainer( index );
      // for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
      //   for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
      //     gradV2 += l * eps * v.myContainer( index_i ) * lap_operator_v.myContainer.coeff( index_i,index_j ) * v.myContainer( index_j );
      
      // l(1-v)^2/4e 
      trace.info() << "- l(1-v)^2/4e" << std::endl;
      double Vm12 = 0.0;
      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
        Vm12 += (l*h/(4*eps)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
      
      // l.per
      trace.info() << "- l.per" << std::endl;
      double Lper = gradV2 + Vm12;
      //      double per = 0.0;
      //      for ( Calculus::Index index_i = 0; index_i < v.myContainer.rows(); index_i++)
      //      {
      //        per += (1/(4*e)) * (1 - 2*v.myContainer( index_i ) + v.myContainer( index_i )*v.myContainer( index_i ));
      //        for ( Calculus::Index index_j = 0; index_j < v.myContainer.rows(); index_j++)
      //            per += e * v.myContainer( index_i ) * tBB.myContainer( index_i,index_j ) * v.myContainer( index_j );
      //      }
      
      // AT tot
      double ATtot = UmG2 + V2gradU2 + gradV2 + Vm12;
      
      //f << "l  " << "  a  " << "  e  " << "  a(u-g)^2  " << "  v^2|grad u|^2  " << "  le|grad v|^2  " << "  l(1-v)^2/4e  " << "  l.per  " << "  AT tot"<< endl;   
      f << tronc(l,8) << "\t" << a << "\t"  << tronc(eps,4) << "\t"  << tronc(UmG2,5) << "\t"  << tronc(V2gradU2,5) << "\t"  << tronc(gradV2,5) << "\t" << tronc(Vm12,5) << "\t" << tronc(Lper,5)  << "\t" << tronc(ATtot,5) << endl; 

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



