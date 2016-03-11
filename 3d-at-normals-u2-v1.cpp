/**
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **/
/**
 * @file 3d-at-normals.cpp
 * @author Jacques-Olivier Lachaud (\c jacques-olivier.lachaud@univ-savoie.fr)
 * LAboratoire de MAthématiques - LAMA (CNRS, UMR 5127), Université de Savoie, France
 *
 * @date 2016/03/10
 *
 * Vol file viewer, with normals regularized by Ambrosio-Tortorelli functionnal
 *
 * Uses IntegralInvariantCurvatureEstimation
 * @see related article:
 *       Coeurjolly, D.; Lachaud, J.O; Levallois, J., (2013). Integral based Curvature
 *       Estimators in Digital Geometry. DGCI 2013. Retrieved from
 *       https://liris.cnrs.fr/publis/?id=5866
 *
 * This file is part of the DGtalTools.
 */

///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include "DGtal/base/Common.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// Shape constructors
#include "DGtal/io/readers/GenericReader.h"
#include "DGtal/images/ImageSelector.h"
#include "DGtal/images/imagesSetsUtils/SetFromImage.h"
#include "DGtal/images/IntervalForegroundPredicate.h"
#include "DGtal/topology/SurfelAdjacency.h"
#include "DGtal/topology/helpers/Surfaces.h"
#include "DGtal/topology/SetOfSurfels.h"
#include "DGtal/topology/CubicalComplex.h"

#include "DGtal/images/ImageHelper.h"
#include "DGtal/topology/DigitalSurface.h"
#include "DGtal/graph/DepthFirstVisitor.h"
#include "DGtal/graph/GraphVisitorRange.h"
#include "DGtal/math/ScalarFunctors.h"
#include "DGtal/geometry/surfaces/estimation/estimationFunctors/ElementaryConvolutionNormalVectorEstimator.h"
#include "DGtal/geometry/surfaces/estimation/LocalEstimatorFromSurfelFunctorAdapter.h"
// Noise
#include "DGtal/geometry/volumes/KanungoNoise.h"

// Integral Invariant includes
#include "DGtal/geometry/surfaces/estimation/IIGeometricFunctors.h"
#include "DGtal/geometry/surfaces/estimation/IntegralInvariantVolumeEstimator.h"
#include "DGtal/geometry/surfaces/estimation/IntegralInvariantCovarianceEstimator.h"

// DEC
#include "DGtal/math/linalg/EigenSupport.h"
#include "DGtal/dec/DiscreteExteriorCalculus.h"
#include "DGtal/dec/DiscreteExteriorCalculusSolver.h"



// Drawing
#include "DGtal/io/boards/Board3D.h"
#include "DGtal/io/colormaps/GradientColorMap.h"

#ifdef WITH_VISU3D_QGLVIEWER
#include "DGtal/io/viewers/Viewer3D.h"
#endif

using namespace DGtal;
using namespace functors;


//-----------------------------------------------------------------------------
// Some useful functions for DEC
//-----------------------------------------------------------------------------

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
//-----------------------------------------------------------------------------



/**
 * Missing parameter error message.
 *
 * @param param
 */
void missingParam( std::string param )
{
  trace.error() << " Parameter: " << param << " is required.";
  trace.info() << std::endl;
}

namespace po = boost::program_options;

int main( int argc, char** argv )
{
  // parse command line ----------------------------------------------
  po::options_description general_opt("Allowed options are");
  general_opt.add_options()
    ("help,h", "display this message")
    ("input,i", po::value< std::string >(), ".vol file")
    ("trivial-radius,t", po::value<double>()->default_value( 3 ), "the parameter t defining the radius for the Trivial estimator, which is used for reorienting II or VCM normal estimations." )
    ("r-radius,r",  po::value< double >(), "Kernel radius r for IntegralInvariant estimator" )
    ("noise,k",  po::value< double >()->default_value(0.5), "Level of Kanungo noise ]0;1[" )
    ("min,l",  po::value<  int >()->default_value(0), "set the minimal image threshold to define the image object (object defined by the voxel with intensity belonging to ]min, max ] )." )
    ("max,u",  po::value<  int >()->default_value(255), "set the minimal image threshold to define the image object (object defined by the voxel with intensity belonging to ]min, max] )." )
    ("lambda,L", po::value<double>()->default_value( 0.05 ), "the parameter lambda of AT functional." )
    ("alpha,a", po::value<double>()->default_value( 0.1 ), "the parameter alpha of AT functional." )
    ("epsilon,e", po::value<double>()->default_value( 4.0 ), "the initial parameter epsilon of AT functional." );

  bool parseOK = true;
  po::variables_map vm;
  try
    {
      po::store( po::parse_command_line( argc, argv, general_opt ), vm );
    }
  catch( const std::exception & ex )
    {
      parseOK = false;
      trace.error() << " Error checking program options: " << ex.what() << std::endl;
    }
  bool neededArgsGiven=true;

  if (parseOK && !(vm.count("input"))){
    missingParam("--input");
    neededArgsGiven=false;
  }
  if (parseOK && !(vm.count("r-radius"))){
    missingParam("--r-radius");
    neededArgsGiven=false;
  }

  double noiseLevel = vm["noise"].as< double >();
  if( noiseLevel < 0.0 || noiseLevel > 1.0 )
    {
      parseOK = false;
      trace.error() << "The noise level should be in the interval: [0, 1]"<< std::endl;
    }

  if(!neededArgsGiven || !parseOK || vm.count("help") || argc <= 1 )
    {
      trace.info()<< "Vol file viewer, with normals regularized by Ambrosio-Tortorelli functionnal" <<std::endl
                  << general_opt << "\n"
                  << "Basic usage: "<<std::endl
                  << "\t at-3d-normals -i file.vol -r 5 --noise 0.1"<<std::endl
                  << std::endl;
      return 0;
    }

  QApplication application(argc,argv);


  int min        =  vm["min"].as<  int >();
  int max        =  vm["max"].as<  int >();
  const double h = 1.0; // not pertinent for now.


  //-----------------------------------------------------------------------------
  // Types.
  typedef Z3i::Space                                 Space;
  typedef Z3i::KSpace                                KSpace;
  typedef Space::RealVector                          RealVector;
  typedef KSpace::SCell                              SCell;
  typedef KSpace::Cell                               Cell;
  typedef KSpace::Surfel                             Surfel;
  typedef Z3i::Domain                                Domain;
  typedef ImageSelector<Domain, unsigned char>::Type Image;
  typedef IntervalForegroundPredicate< Image >       Object;
  typedef KanungoNoise< Object, Domain >             KanungoPredicate;
  typedef BinaryPointPredicate<DomainPredicate<Domain>, KanungoPredicate, AndBoolFct2  > NoisyObject;
  typedef KSpace::SurfelSet                          SurfelSet;
  typedef SetOfSurfels< KSpace, SurfelSet >          MySetOfSurfels;
  typedef DigitalSurface< MySetOfSurfels >           MyDigitalSurface;
  typedef MyDigitalSurface::ConstIterator            ConstIterator;

  //-----------------------------------------------------------------------------
  // Loading vol file.
  trace.beginBlock( "Loading vol file." );
  std::string inputFile = vm[ "input" ].as< std::string >();
  std::string extension = inputFile.substr(inputFile.find_last_of(".") + 1);
  if(extension!="vol" && extension != "p3d" && extension != "pgm3D" && extension != "pgm3d" && extension != "sdp" && extension != "pgm" )
    {
    trace.info() << "File extension not recognized: "<< extension << std::endl;
    return 0;
  }
  Image image = GenericReader<Image>::import (inputFile );
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Extracting object with possible noise.
  trace.beginBlock( "Extracting object with possible noise." );
  Object                  object( image,  min, max );
  KanungoPredicate        kanungo_pred( object, image.domain(), noiseLevel );
  DomainPredicate<Domain> domain_pred( image.domain() );
  AndBoolFct2             andF;
  NoisyObject             noisy_object(domain_pred, kanungo_pred, andF  );
  Domain                  domain = image.domain();
  KSpace                  K;
  bool space_ok = K.init( domain.lowerBound()-Z3i::Domain::Point::diagonal(),
                          domain.upperBound()+Z3i::Domain::Point::diagonal(), true );
  if (!space_ok)
    {
      trace.error() << "Error in the Khalimsky space construction."<<std::endl;
      return 2;
    }
  CanonicSCellEmbedder< KSpace >       embedder( K );
  SurfelAdjacency< KSpace::dimension > surfAdj( true );
  trace.endBlock();

  //-----------------------------------------------------------------------------
  //! [3dVolBoundaryViewer-ExtractingSurface]
  trace.beginBlock( "Extracting boundary by scanning the space. " );
  MySetOfSurfels theSetOfSurfels( K, surfAdj );
  Surfaces<KSpace>::sMakeBoundary( theSetOfSurfels.surfelSet(),
                                   K, image,
                                   domain.lowerBound(),
                                   domain.upperBound() );
  MyDigitalSurface digSurf( theSetOfSurfels );
  trace.info() << "Digital surface has " << digSurf.size() << " surfels."
               << std::endl;
  trace.endBlock();
  //! [3dVolBoundaryViewer-ExtractingSurface]

  // Map surfel -> estimated normals.
  std::map<SCell,RealVector> n_estimations;

  //-----------------------------------------------------------------------------
  // Estimating orientation of normals
  trace.beginBlock( "Estimating orientation of normals by simple convolutions of trivial surfel normals." );
  double t = vm["trivial-radius"].as<double>();
  typedef RealVector::Component                    Scalar;
  typedef functors::HatFunction<Scalar>            Functor;
  typedef functors::ElementaryConvolutionNormalVectorEstimator< Surfel, CanonicSCellEmbedder<KSpace> > SurfelFunctor;
  typedef ExactPredicateLpSeparableMetric<Space,2> Metric;
  typedef LocalEstimatorFromSurfelFunctorAdapter< MySetOfSurfels, Metric, SurfelFunctor, Functor>      NormalEstimator;
  Functor                      fct( 1.0, t );
  CanonicSCellEmbedder<KSpace> canonic_embedder( K );
  SurfelFunctor                surfelFct( canonic_embedder, 1.0 );
  NormalEstimator              nt_estimator;
  Metric                       aMetric;
  std::vector<RealVector>      nt_estimations;
  nt_estimator.attach( digSurf );
  nt_estimator.setParams( aMetric, surfelFct, fct, t );
  nt_estimator.init( h, digSurf.begin(), digSurf.end());
  nt_estimator.eval( digSurf.begin(), digSurf.end(), std::back_inserter( nt_estimations ) );
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Estimating normals
  trace.beginBlock( "Estimating normals with II." );
  typedef typename Domain::ConstIterator DomainConstIterator;
  typedef functors::IINormalDirectionFunctor<Space> IINormalFunctor;
  typedef IntegralInvariantCovarianceEstimator<KSpace, NoisyObject, IINormalFunctor> IINormalEstimator;
  std::vector<RealVector> nii_estimations;
  const double            r = vm["r-radius"].as<double>();
  IINormalEstimator       nii_estimator( K, noisy_object );
  trace.info() << " r=" << r << std::endl;
  nii_estimator.setParams( r );
  nii_estimator.init( h, digSurf.begin(), digSurf.end() );
  nii_estimator.eval( digSurf.begin(), digSurf.end(), std::back_inserter( nii_estimations ) );
  // Fix orientations of ii.
  for ( unsigned int i = 0; i < nii_estimations.size(); ++i )
    {
      if ( nii_estimations[ i ].dot( nt_estimations[ i ] ) < 0.0 )
        nii_estimations[ i ] *= -1.0;
    }
  trace.info() << "- nb estimations  = " << nii_estimations.size() << std::endl;
  trace.endBlock();

  // The chosen estimator is II.
  {
    unsigned int i = 0;
    for ( ConstIterator it = digSurf.begin(), itE = digSurf.end(); it != itE; ++it, ++i )
      {
        RealVector nii = nii_estimations[ i ];
        nii /= nii.norm();
        n_estimations[ *it ] = nii;
      }
  }

  //-----------------------------------------------------------------------------
  //! [3dVolBoundaryViewer-ViewingSurface]
  trace.beginBlock( "Displaying everything. " );
  Viewer3D<Space,KSpace> viewer( K );
  viewer.setWindowTitle("Simple boundary of volume Viewer");
  viewer.show();
  viewer << SetMode3D(K.unsigns( *(digSurf.begin()) ).className(), "Basic");
  unsigned int i = 0;
  for ( ConstIterator it = digSurf.begin(), itE = digSurf.end(); it != itE; ++it, ++i )
    {
      viewer.setFillColor( Color( 200, 200, 250 ) );
      Display3DFactory<Space,KSpace>::drawOrientedSurfelWithNormal( viewer, *it, n_estimations[ *it ], false );
    }
  viewer << Viewer3D<>::updateDisplay;
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Defining Discrete Calculus.
  typedef CubicalComplex< KSpace >                                 CComplex;
  typedef DiscreteExteriorCalculus<2,3, EigenLinearAlgebraBackend> Calculus;
  typedef Calculus::Index                                          Index;
  typedef Calculus::PrimalForm0                                    PrimalForm0;
  typedef Calculus::PrimalForm1                                    PrimalForm1;
  typedef Calculus::PrimalForm2                                    PrimalForm2;
  typedef Calculus::PrimalIdentity0                                PrimalIdentity0;
  typedef Calculus::PrimalIdentity1                                PrimalIdentity1;
  typedef Calculus::PrimalIdentity2                                PrimalIdentity2;
  trace.beginBlock( "Creating Discrete Exterior Calculus. " );
  Calculus calculus;
  calculus.initKSpace<Domain>( domain );
  const KSpace& Kc = calculus.myKSpace; // should not be used.
  // Use a cubical complex to find all lower incident cells easily.
  CComplex complex( K );
  for ( ConstIterator it = digSurf.begin(), itE = digSurf.end(); it != itE; ++it )
    complex.insertCell( 2, K.unsigns( *it ) );
  complex.close();
  for ( CComplex::CellMapIterator it = complex.begin( 0 ), itE = complex.end( 0 ); it != itE; ++it )
    calculus.insertSCell( K.signs( it->first, K.POS ) );
  
  for ( CComplex::CellMapIterator it = complex.begin( 1 ), itE = complex.end( 1 ); it != itE; ++it )
    {
      SCell     linel = K.signs( it->first, K.POS );
      Dimension k     = * K.sDirs( linel );
      bool      pos   = K.sDirect( linel, k );
      calculus.insertSCell( pos ? linel : K.sOpp( linel ) );
      // calculus.insertSCell( K.signs( it->first, K.POS ) );
    }

  // for ( CComplex::CellMapIterator it = complex.begin( 2 ), itE = complex.end( 2 ); it != itE; ++it )
  // calculus.insertSCell( K.signs( it->first, K.POS ) );
  for ( ConstIterator it = digSurf.begin(), itE = digSurf.end(); it != itE; ++it )
    {
      calculus.insertSCell( *it );
      // SCell     surfel = *it;
      // Dimension k      = K.sOrthDir( surfel );
      // bool      pos    = K.sDirect( surfel, k );
      // calculus.insertSCell( pos ? *it : K.sOpp( *it ) );
    }
  calculus.updateIndexes();
  trace.info() << calculus << endl;

  std::vector<PrimalForm2> g;
  g.reserve( 3 );
  g.push_back( PrimalForm2( calculus ) );
  g.push_back( PrimalForm2( calculus ) );
  g.push_back( PrimalForm2( calculus ) );
  Index nb2 = g[ 0 ].myContainer.rows();
  
  for ( Index index = 0; index < nb2; index++)
    {
      const Calculus::SCell& cell = g[ 0 ].getSCell( index );
      if ( theSetOfSurfels.isInside( cell ) ) 
        {
          const RealVector&      n    = n_estimations[ cell ];
          g[ 0 ].myContainer( index ) = n[ 0 ];
          g[ 1 ].myContainer( index ) = n[ 1 ];
          g[ 2 ].myContainer( index ) = n[ 2 ];
        }
      else
        {
          const RealVector&      n    = n_estimations[ K.sOpp( cell ) ];
          g[ 0 ].myContainer( index ) = n[ 0 ];
          g[ 1 ].myContainer( index ) = n[ 1 ];
          g[ 2 ].myContainer( index ) = n[ 2 ];
        }
    }
  cout << endl;
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
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Building AT functional.
  trace.beginBlock( "Building AT functional. " );
  double a  = vm[ "alpha" ].as<double>();
  double e  = vm[ "epsilon" ].as<double>();
  double l  = vm[ "lambda" ].as<double>();

  // u = g at the beginning
  trace.info() << "u[0,1,2]" << endl;
  std::vector<PrimalForm2> u;
  u.push_back( g[ 0 ] ); u.push_back( g[ 1 ] ); u.push_back( g[ 2 ] );
  // v = 1 at the beginning
  trace.info() << "v" << endl;
  PrimalForm1 v( calculus );
  Index nb1 = v.myContainer.rows();
  for ( Index index = 0; index < nb1; index++)  v.myContainer( index ) = 1.0;
  const PrimalIdentity0 Id0 = calculus.identity<0, PRIMAL>();
  const PrimalIdentity1 Id1 = calculus.identity<1, PRIMAL>();
  const PrimalIdentity2 Id2 = calculus.identity<2, PRIMAL>();
  // Building alpha_
  trace.info() << "alpha_g" << endl;
  const PrimalIdentity2 alpha_Id2 = a * Id2; // a * invG0;
  vector<PrimalForm2> alpha_g;
  alpha_g.push_back( alpha_Id2 * g[ 0 ] );
  alpha_g.push_back( alpha_Id2 * g[ 1 ] );
  alpha_g.push_back( alpha_Id2 * g[ 2 ] );
  trace.info() << "lap_operator_v" << endl;
  const PrimalIdentity1 lap_operator_v = -1.0 * ( primal_D0 * dual_h2 * dual_D1 * primal_h1 
                                                  + dual_h1 * dual_D0 * primal_h2 * primal_D1 );
  // SparseLU is so much faster than SparseQR
  // SimplicialLLT is much faster than SparseLU
  // typedef EigenLinearAlgebraBackend::SolverSparseQR LinearAlgebraSolver;
  // typedef EigenLinearAlgebraBackend::SolverSparseLU LinearAlgebraSolver;
  typedef EigenLinearAlgebraBackend::SolverSimplicialLLT LinearAlgebraSolver;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 2, PRIMAL, 2, PRIMAL> SolverU;
  SolverU solver_u;
  typedef DiscreteExteriorCalculusSolver<Calculus, LinearAlgebraSolver, 1, PRIMAL, 1, PRIMAL> SolverV;
  SolverV solver_v;

  trace.info() << "lB'B'" << endl;
  const PrimalIdentity1 lBB = l * lap_operator_v;
  PrimalForm1 l_sur_4( calculus );
  for ( Index index = 0; index < nb1; index++)
    l_sur_4.myContainer( index ) = l / 4.0;
  double coef_eps = 2.0;
  double eps      = 2.0 * e;
  const int n     = 10;
  trace.endBlock();
      
  //-----------------------------------------------------------------------------
  // Solving AT functional.
  trace.beginBlock( "Solving AT functional. " );
  while ( eps / coef_eps >= h )
    {
      eps /= coef_eps;
      trace.info() << "************** epsilon = " << eps << "***************************************" << endl;
      const PrimalIdentity1 BB = eps * lBB + ( l/(4.0*eps) ) * Id1; // tS_S;
      for ( int i = 0; i < n; ++i )
        {
          trace.info() << "---------- Iteration " << i << "/" << n << " ------------------------------" << endl;
          trace.info() << "-------------------------------------------------------------------------------" << endl;
          trace.beginBlock("Solving for u");
          trace.info() << "Building matrix Av2A" << endl;
          PrimalIdentity1 diag_v  = diag( calculus, v );
          PrimalIdentity2 U_Id2 = -1.0 * primal_D1 * diag_v * diag_v * dual_h1 * dual_D0 * primal_h2
            + alpha_Id2;
          trace.info() << "Prefactoring matrix Av2A + alpha_iG0" << endl;
          solver_u.compute( U_Id2 );
          for ( unsigned int d = 0; d < 3; ++d )
            {
              trace.info() << "Solving (Av2A + alpha_iG0) u[" << d << "] = ag[" << d << "]" << endl;
              u[ d ] = solver_u.solve( alpha_g[ d ] );
              trace.info() << "  => " << ( solver_u.isValid() ? "OK" : "ERROR" ) 
                           << " " << solver_u.myLinearAlgebraSolver.info() << endl;
            }
          trace.info() << "-------------------------------------------------------------------------------" << endl;
          trace.endBlock();

          trace.beginBlock("Solving for v");
          const PrimalForm1 former_v = v;
          trace.info() << "Building matrix tu_tA_A_u + BB + Mw2" << endl;
          PrimalIdentity1 V_Id1 = BB;
          for ( unsigned int d = 0; d < 3; ++d )
            {
              const PrimalIdentity1 A_u = diag( calculus, dual_h1 * dual_D0 * primal_h2 * u[ d ] );
              V_Id1.myContainer += square( calculus, A_u ).myContainer;
            }
          trace.info() << "Prefactoring matrix tu_tA_A_u + BB + Mw2" << endl;
          solver_v.compute( V_Id1 );
          trace.info() << "Solving (tu_tA_A_u + BB + Mw2) v = 1/(4eps) * l" << endl;
          v = solver_v.solve( (1.0/eps) * l_sur_4 );
          trace.info() << "  => " << ( solver_v.isValid() ? "OK" : "ERROR" ) 
                       << " " << solver_v.myLinearAlgebraSolver.info() << endl;
          trace.info() << "-------------------------------------------------------------------------------" << endl;
          trace.endBlock();

          for ( Index index = 0; index < nb2; index++)
            {
              double n2 = 0.0;
              for ( unsigned int d = 0; d < 3; ++d )
                n2 += u[ d ].myContainer( index ) * u[ d ].myContainer( index );
              double norm = sqrt( n2 );
              for ( unsigned int d = 0; d < 3; ++d )
                u[ d ].myContainer( index ) /= norm;
            }

          trace.beginBlock("Checking v, computing norms");
          double m1 = 1.0;
          double m2 = 0.0;
          double ma = 0.0;
          for ( Index index = 0; index < nb1; index++)
            {
              double val = v.myContainer( index );
              m1 = std::min( m1, val );
              m2 = std::max( m2, val );
              ma += val;
            }
          trace.info() << "1-form v: min=" << m1 << " avg=" << ( ma / nb1 ) << " max=" << m2 << std::endl;
          for ( Index index = 0; index < nb1; index++)
            v.myContainer( index ) = std::min( std::max(v.myContainer( index ), 0.0) , 1.0 );
          double n_infty = 0.0;
          double n_2 = 0.0;
          double n_1 = 0.0;
          for ( Index index = 0; index < nb1; index++)
            {
              n_infty = std::max( n_infty, fabs( v.myContainer( index ) - former_v.myContainer( index ) ) );
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
        } // for ( int i = 0; i < n; ++i )
    }
  trace.endBlock();

  //-----------------------------------------------------------------------------
  // Displaying regularized normals
  trace.beginBlock( "Displaying regularized normals. " );
  Viewer3D<Space,KSpace> viewerR( K );
  viewerR.setWindowTitle("Regularized normals");
  viewerR.show();
  viewerR << SetMode3D(K.unsigns( *(digSurf.begin()) ).className(), "Basic");
  viewerR.setFillColor( Color( 200, 200, 250 ) );
  for ( Index index = 0; index < nb2; index++)
    {
      const SCell& cell    = u[ 0 ].getSCell( index );
      // const RealVector& n  = n_estimations[ cell ];
      RealVector nr        = RealVector( u[ 0 ].myContainer( index ), 
                                         u[ 1 ].myContainer( index ), 
                                         u[ 2 ].myContainer( index ) );
      nr /= nr.norm();
      if ( theSetOfSurfels.isInside( cell ) ) 
        Display3DFactory<Space,KSpace>::drawOrientedSurfelWithNormal( viewerR, cell, nr, false );
      else
        Display3DFactory<Space,KSpace>::drawOrientedSurfelWithNormal( viewerR, K.sOpp( cell ), nr, false );
    }
  viewerR.setLineColor( Color( 255, 0, 0 ) );
  for ( Index index = 0; index < nb1; index++)
    {
      const SCell& cell    = v.getSCell( index );
      Dimension    k       = * K.sDirs( cell ); 
      const SCell  p0      = K.sIncident( cell, k, true );
      const SCell  p1      = K.sIncident( cell, k, false );
      if ( v.myContainer( index ) >= 0.5 ) continue;
      viewerR.addLine( embedder.embed( p0 ), embedder.embed( p1 ), (0.5 - v.myContainer( index ))/ 5.0 );
    }
  viewerR << Viewer3D<>::updateDisplay;
  trace.endBlock();

  return application.exec();

}
