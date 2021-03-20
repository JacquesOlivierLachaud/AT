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

#pragma once

/**
 * @file Clock.h
 * @author Jacques-Olivier Lachaud (\c jacques-olivier.lachaud@univ-savoie.fr )
 * Laboratory of Mathematics (CNRS, UMR 5807), University of Savoie, France
 * @author David Coeurjolly (\c david.coeurjolly@liris.cnrs.fr )
 * Laboratoire d'InfoRmatique en Image et Systèmes d'information - LIRIS (CNRS, UMR 5205), CNRS, France
 *
 * @date 2009/12/11
 *
 * Header file for module Clock.cpp
 *
 * This file is part of the DGtal library (backported from Imagene)
 */

#ifdef Clock_RECURSES
#error Recursive header files inclusion detected in Clock.h
#else // defined(Clock_RECURSES)
/** Prevents recursive inclusion of headers. */
#define Clock_RECURSES

#ifndef Clock_h
/** Prevents repeated inclusion of headers. */
#define Clock_h

//////////////////////////////////////////////////////////////////////////////
// Inclusions
#include <iostream>
#include <cstdlib>

#if ( (defined(UNIX)||defined(unix)||defined(linux)) )
#include <sys/time.h>
#include <time.h>
#endif

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifdef WIN32
#include <time.h>
#endif



//////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// class Clock
/**
 * Description of class 'Clock' <p>
 * Aim: To provide functions to start and stop a timer. Is useful to get
 * performance of algorithms.
 *
 * The following code snippet demonstrates how to use \p Clock
 *
 *  \code
 *  #include "Clock.h"
 *
 *   Clock c;
 *   long duration;
 *
 *   c.startClock();
 *   ...
 *   //do something
 *   ...
 *   duration = c.stopClock();
 *
 *   std::cout<< "Duration in ms. : "<< duration <<endl;
 *  \endcode
 *
 * @see testClock.cpp
 */
class Clock
{
  // ----------------------- Standard services ------------------------------
  // -------------------------- timing services -------------------------------
public:
  /**
   * Starts a clock.
   */
  inline void startClock()
  {
    
#ifdef WIN32
    myFirstTick = clock();
    if (myFirstTick == (clock_t) -1)
      {
        std::cerr << "[Clock::startClock] Error: can't start clock." << std::endl;
      }
#else
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    myTimerStart.tv_sec = mts.tv_sec;
    myTimerStart.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &myTimerStart);
#endif
#endif
  }

  /**
   * Stops the clock.
   * @return the time (in ms) since the last 'startClock()' or 'restartClock()'.
   */
  inline double stopClock() const
  {
    
#ifdef WIN32
    clock_t last_tick = clock();
    if (last_tick == (clock_t) -1)
      {
        std::cerr << "[Clock::stopClock] Error: can't stop clock." << std::endl;
      }
    return (double) ((double) 1000.0 * (double)(last_tick - myFirstTick)
                     / (double) CLOCKS_PER_SEC);
#else
    struct timespec current;
    
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    current.tv_sec = mts.tv_sec;
    current.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &current); //Linux gettime
#endif
    
    return (( current.tv_sec - myTimerStart.tv_sec) *1000 +
            ( current.tv_nsec - myTimerStart.tv_nsec)/1000000.0);
#endif
    
  }

  /**
   * Restart the clock.
   * @return the time (in ms) since the last 'startClock()' or 'restartClock()'.
   */
  double restartClock()
  {
    
#ifdef WIN32
    clock_t last_tick = clock();
    if (last_tick == (clock_t) -1)
      {
        std::cerr << "[Clock::stopClock] Error: can't restart clock." << std::endl;
      }
    const double delta = ((double) 1000.0 * (double)(last_tick - myFirstTick)
                          / (double) CLOCKS_PER_SEC);
    myFirstTick = last_tick;
    return delta;
#else
    struct timespec current;
    
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    current.tv_sec = mts.tv_sec;
    current.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &current); //Linux gettime
#endif
    
    const double delta = (( current.tv_sec - myTimerStart.tv_sec) *1000 +
                          ( current.tv_nsec - myTimerStart.tv_nsec)/1000000.0);
    myTimerStart.tv_sec = current.tv_sec;
    myTimerStart.tv_nsec = current.tv_nsec;
    return delta;
#endif
    
  }
  
  /**
   * Constructor.
   *
   */
  Clock() = default;

  /**
   * Destructor.
   */
  ~Clock() = default;


  // ------------------------- Private Datas --------------------------------
private:

  ///internal timer object;
#ifdef WIN32
  clock_t myFirstTick;
#else
  struct timespec myTimerStart;
#endif

}; // end of class Clock


#endif // !defined Clock_h

#undef Clock_RECURSES
#endif // else defined(Clock_RECURSES)
