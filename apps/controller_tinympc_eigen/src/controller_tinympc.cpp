/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * controller_tinympc.c - App layer application of TinyMPC.
 */

/** 
 * Single lap
 */

#include "Eigen.h"
using namespace Eigen;

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "app.h"
#include "config.h"
#include "FreeRTOS.h"
#include "task.h"

#include "controller.h"
#include "physicalConstants.h"
#include "log.h"
#include "param.h"
#include "num.h"
#include "math3d.h"
#include "stabilizer_types.h"  // For controlModePWM

#include "cpp_compat.h"   // needed to compile Cpp to C

#include "tinympc/tinympc.h"
#define TINYMPC_TASK_STACKSIZE        (3 * configMINIMAL_STACK_SIZE)

// Rodriguez parameters conversion function (needed for old firmware compatibility)
static inline struct vec quat2rp(struct quat q) {
  struct vec v;
  float w_abs = fabsf(q.w);
  if (w_abs > 1e-6f) { // Avoid division by near-zero
    v.x = q.x / q.w;
    v.y = q.y / q.w;
    v.z = q.z / q.w;
  } else {
    // Handle singular case
    v.x = 0.0f;
    v.y = 0.0f;
    v.z = 0.0f;
  }
  return v;
}

// Edit the debug name to get nice debug prints
#define DEBUG_MODULE "TINYMPC-E"
#include "debug.h"

// Benchmark-only mode: run synthetic MPC solves in init and keep motors off.
#define TINYMPC_BENCH_ONLY 1
#define BENCH_TARGET_RATE_HZ 500
#define BENCH_SOLVES_PER_POINT 80
#define BENCH_CHUNK_SIZE 1
#define BENCH_CHUNK_DELAY_MS 200
#define BENCH_PROGRESS_EVERY 0
#define BENCH_H_SWEEP_POINTS 4
#define BENCH_I_SWEEP_POINTS 8

static bool benchRequested;
static bool benchDone;
static void runSyntheticBenchmarkChunk(const uint32_t chunkSize);

void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) {
#if TINYMPC_BENCH_ONLY
    if (benchRequested && !benchDone) {
      runSyntheticBenchmarkChunk(BENCH_CHUNK_SIZE);
      vTaskDelay(M2T(BENCH_CHUNK_DELAY_MS));
      continue;
    }
#endif
    vTaskDelay(M2T(200));
  }
}

// Macro variables - define locally to avoid dependency issues
#define DT 0.002f       // dt
#define NHORIZON 50     // horizon steps (must match constants.h if used)
#define MPC_RATE RATE_100_HZ  // control frequency
#define LQR_RATE RATE_500_HZ  // control frequency

/* Include trajectory to track */
#include "traj_fig8_12.h"
// #include "traj_circle_500hz.h"  // Large circle (1m radius)
// #include "traj_circle_small.h"  // Small circle (0.5m radius)
// #include "traj_perching.h"
//#include "traj_straight_line.h"  // Straight line (0,0,0.5) to (1,0,0.5)

// Precomputed data and cache, in params_*.h
static MatrixNf A;
static MatrixNMf B;
static MatrixMNf Kinf;
static MatrixMNf Klqr;
static MatrixNf Pinf;
static MatrixMf Quu_inv;
static MatrixNf AmBKt;
static MatrixNMf coeff_d2p;
static MatrixNf Q;
static MatrixMf R;

/* Allocate global variables for MPC */

static VectorNf Xhrz[NHORIZON];
static VectorMf Uhrz[NHORIZON-1]; 
static VectorMf Ulqr;
static VectorMf d[NHORIZON-1];
static VectorNf p[NHORIZON];
static VectorMf YU[NHORIZON];

static VectorNf q[NHORIZON-1];
static VectorMf r[NHORIZON-1];
static VectorMf r_tilde[NHORIZON-1];

static VectorNf Xref[NHORIZON];
static VectorMf Uref[NHORIZON-1];

static MatrixMf Acu;
static VectorMf ucu;
static VectorMf lcu;

static VectorMf Qu;
static VectorMf ZU[NHORIZON-1]; 
static VectorMf ZU_new[NHORIZON-1];

static VectorNf x0;
static VectorNf xg;
static VectorMf ug;

// Create TinyMPC struct
static tiny_Model model;
static tiny_AdmmSettings stgs;
static tiny_AdmmData data;
static tiny_AdmmInfo info;
static tiny_AdmmSolution soln;
static tiny_AdmmWorkspace work;

// Helper variables
static uint64_t startTimestamp;
// static bool isInit = false;  // fix for tracking problem - UNUSED, commented out
// static uint32_t mpcTime = 0;  // UNUSED (was for logging), commented out
static float u_hover[4] = {0.7f, 0.663f, 0.7373f, 0.633f};  // cf1
// static float u_hover[4] = {0.7467, 0.667f, 0.78, 0.7f};  // cf2 not correct
static int8_t result = 0;
static uint32_t step = 0;
static bool en_traj = true;
static uint32_t traj_length = T_ARRAY_SIZE(X_ref_data);
//static int8_t user_traj_iter = 1;  // number of times to execute full trajectory
static int8_t traj_hold = 1;       // hold current trajectory for this no of steps
static int8_t traj_iter = 0;
static uint32_t traj_idx = 0;

static struct vec desired_rpy;
static struct quat attitude;
static struct vec phi;

// Basic mode - no obstacle avoidance constraints

static const int benchHSweep[BENCH_H_SWEEP_POINTS] = {10, 20, 30, 40};
static const int benchIterSweep[BENCH_I_SWEEP_POINTS] = {1, 2, 4, 7, 10, 14, 20, 28};

static int benchHPoint = 0;
static int benchIPoint = 0;
static uint32_t benchIterInPoint = 0;

static uint32_t benchMinUs[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchMaxUs[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint64_t benchSumUs[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchMisses[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchSolved[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchMaxIterReached[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchNonCvx[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];
static uint32_t benchOtherStatus[BENCH_H_SWEEP_POINTS][BENCH_I_SWEEP_POINTS];

static void runSyntheticBenchmarkChunk(const uint32_t chunkSize) {
  const uint32_t budget_us = 1000000U / BENCH_TARGET_RATE_HZ;
  uint32_t chunks = 0;

  while (chunks < chunkSize && benchHPoint < BENCH_H_SWEEP_POINTS) {
    const int effH = benchHSweep[benchHPoint];
    if (effH > NHORIZON) {
      benchHPoint++;
      benchIPoint = 0;
      benchIterInPoint = 0;
      continue;
    }

    const uint32_t k = (uint32_t)(benchHPoint * 100000 + benchIPoint * 10000) + benchIterInPoint;
    stgs.max_iter = benchIterSweep[benchIPoint];
    model.nhorizon = effH;

    for (int i = 0; i < effH; ++i) {
      const float t = (float)(k + (uint32_t)i) * DT;

      Xref[i].setZero();
      Xref[i](0) = 0.8f * sinf(1.2f * t);
      Xref[i](1) = 0.6f * cosf(0.9f * t);
      Xref[i](2) = 0.5f + 0.15f * sinf(0.7f * t);
      Xref[i](6) = 0.8f * 1.2f * cosf(1.2f * t);
      Xref[i](7) = -0.6f * 0.9f * sinf(0.9f * t);
      Xref[i](8) = 0.15f * 0.7f * cosf(0.7f * t);
      Xref[i](11) = 0.2f * sinf(0.5f * t);

      if (i < effH - 1) {
        Uref[i].setZero();
      }
    }

    // Keep warm-start effective while avoiding a trivial fixed-point solve.
    x0 = Xref[0];
    x0(0) += 0.08f * sinf(0.13f * (float)k);
    x0(1) += 0.06f * cosf(0.11f * (float)k);
    x0(2) += 0.03f * sinf(0.07f * (float)k);
    x0(6) += 0.10f * cosf(0.17f * (float)k);
    x0(7) += 0.10f * sinf(0.19f * (float)k);

    tiny_SetInitialState(&work, &x0);
    tiny_SetStateReference(&work, Xref);
    tiny_SetInputReference(&work, Uref);
    tiny_UpdateLinearCost(&work);

    const uint32_t t0 = usecTimestamp();
    tiny_SolveAdmm(&work);
    const uint32_t dt_us = usecTimestamp() - t0;

    if (dt_us < benchMinUs[benchHPoint][benchIPoint]) {
      benchMinUs[benchHPoint][benchIPoint] = dt_us;
    }
    if (dt_us > benchMaxUs[benchHPoint][benchIPoint]) {
      benchMaxUs[benchHPoint][benchIPoint] = dt_us;
    }
    benchSumUs[benchHPoint][benchIPoint] += dt_us;
    if (dt_us > budget_us) {
      benchMisses[benchHPoint][benchIPoint]++;
    }
    if (info.status_val == TINY_SOLVED) {
      benchSolved[benchHPoint][benchIPoint]++;
    } else if (info.status_val == TINY_MAX_ITER_REACHED) {
      benchMaxIterReached[benchHPoint][benchIPoint]++;
    } else if (info.status_val == TINY_NON_CVX) {
      benchNonCvx[benchHPoint][benchIPoint]++;
    } else {
      benchOtherStatus[benchHPoint][benchIPoint]++;
    }

    benchIterInPoint++;
    chunks++;

    if (BENCH_PROGRESS_EVERY > 0 && (benchIterInPoint % BENCH_PROGRESS_EVERY) == 0) {
      DEBUG_PRINT("BENCH progress: hidx=%d/%d H=%d iidx=%d/%d iter=%d done=%lu/%d\n",
                  benchHPoint + 1, BENCH_H_SWEEP_POINTS, effH,
                  benchIPoint + 1, BENCH_I_SWEEP_POINTS, stgs.max_iter,
                  (unsigned long)benchIterInPoint, BENCH_SOLVES_PER_POINT);
    }

    if (benchIterInPoint >= BENCH_SOLVES_PER_POINT) {
      const uint32_t avg_us = (uint32_t)(benchSumUs[benchHPoint][benchIPoint] / BENCH_SOLVES_PER_POINT);
      DEBUG_PRINT("BENCH_CSV,%d,%d,%lu,%lu,%lu,%lu,%d,%lu,%lu,%lu,%lu\n",
                  effH, benchIterSweep[benchIPoint],
                  (unsigned long)benchMinUs[benchHPoint][benchIPoint],
                  (unsigned long)avg_us,
                  (unsigned long)benchMaxUs[benchHPoint][benchIPoint],
                  (unsigned long)benchMisses[benchHPoint][benchIPoint],
                  BENCH_SOLVES_PER_POINT,
                  (unsigned long)benchSolved[benchHPoint][benchIPoint],
                  (unsigned long)benchMaxIterReached[benchHPoint][benchIPoint],
                  (unsigned long)benchNonCvx[benchHPoint][benchIPoint],
                  (unsigned long)benchOtherStatus[benchHPoint][benchIPoint]);

      benchIPoint++;
      benchIterInPoint = 0;
      if (benchIPoint >= BENCH_I_SWEEP_POINTS) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        for (int p = 0; p < BENCH_I_SWEEP_POINTS; ++p) {
          const double x = (double)benchIterSweep[p];
          const double y = (double)(benchSumUs[benchHPoint][p] / BENCH_SOLVES_PER_POINT);
          sum_x += x;
          sum_y += y;
          sum_xx += x * x;
          sum_xy += x * y;
        }
        const double n = (double)BENCH_I_SWEEP_POINTS;
        const double denom = n * sum_xx - sum_x * sum_x;
        double k_iter = 0.0;
        double t0 = 0.0;
        if (fabs(denom) > 1e-9) {
          k_iter = (n * sum_xy - sum_x * sum_y) / denom;
          t0 = (sum_y - k_iter * sum_x) / n;
        }
        DEBUG_PRINT("BENCH_FIT_CSV,%d,%.3f,%.3f,%.6f\n",
                    effH, t0, k_iter, k_iter / (double)effH);
        benchHPoint++;
        benchIPoint = 0;
      }
    }
  }

  if (benchHPoint >= BENCH_H_SWEEP_POINTS) {
    benchDone = true;
  }
}

void updateInitialState(const sensorData_t *sensors, const state_t *state) {
  x0(0) = state->position.x;
  x0(1) = state->position.y;
  x0(2) = state->position.z;
  // Body velocity error, [m/s]                          
  x0(6) = state->velocity.x;
  x0(7) = state->velocity.y;
  x0(8) = state->velocity.z;
  // Angular rate error, [rad/s]
  x0(9)  = radians(sensors->gyro.x);   
  x0(10) = radians(sensors->gyro.y);
  x0(11) = radians(sensors->gyro.z);
  attitude = mkquat(
    state->attitudeQuaternion.x,
    state->attitudeQuaternion.y,
    state->attitudeQuaternion.z,
    state->attitudeQuaternion.w);  // current attitude
  phi = quat2rp(qnormalize(attitude));  // quaternion to Rodriquez parameters  
  // Attitude error
  x0(3) = phi.x;
  x0(4) = phi.y;
  x0(5) = phi.z;
}

void updateHorizonReference(const setpoint_t *setpoint) {
  // Update reference: from stored trajectory or commander
  if (en_traj) {
    if (step % traj_hold == 0) {
      traj_idx = (int)(step / traj_hold);
      for (int i = 0; i < NHORIZON; ++i) {
        for (int j = 0; j < NSTATES; ++j) {
          Xref[i](j) = X_ref_data[traj_idx][j];
        }
        if (i < NHORIZON - 1) {
          for (int j = 0; j < NINPUTS; ++j) {
            Uref[i](j) = U_ref_data[traj_idx][j];
          }          
        }
      }
    }
  }
  else {
    xg(0)  = setpoint->position.x;
    xg(1)  = setpoint->position.y;
    xg(2)  = setpoint->position.z;
    xg(6)  = setpoint->velocity.x;
    xg(7)  = setpoint->velocity.y;
    xg(8)  = setpoint->velocity.z;
    xg(9)  = radians(setpoint->attitudeRate.roll);
    xg(10) = radians(setpoint->attitudeRate.pitch);
    xg(11) = radians(setpoint->attitudeRate.yaw);
    desired_rpy = mkvec(radians(setpoint->attitude.roll), 
                        radians(setpoint->attitude.pitch), 
                        radians(setpoint->attitude.yaw));
    attitude = rpy2quat(desired_rpy);
    phi = quat2rp(qnormalize(attitude));  
    xg(3) = phi.x;
    xg(4) = phi.y;
    xg(5) = phi.z;
    tiny_SetGoalState(&work, Xref, &xg);
    tiny_SetGoalInput(&work, Uref, &ug);
    // // xg(1) = 1.0;
    // // xg(2) = 2.0;
  }
  // DEBUG_PRINT("z_ref = %.2f\n", (double)(Xref[0](2)));

  // Trajectory progression
  if (en_traj) {
    if (traj_idx >= traj_length - 1 - NHORIZON + 1) { 
      // Reached end of trajectory - hold at final position
      // Don't reset step, just stay at the end
    } 
    else {
      step += 1;
    }
  }
}

// Half-space constraint function removed for basic functionality test

void controllerOutOfTreeInit(void) { 
  /* Start MPC initialization*/

  // Precompute/Cache
  // #include "params_500hz.h"
  #include "params_100hz.h"  // Original gains (stable)
  // #include "params_constrained.h"

  // End of Precompute/Cache

  tiny_InitModel(&model, NSTATES, NINPUTS, NHORIZON, 0, 0, DT, &A, &B, 0);
  tiny_InitSettings(&stgs);
  stgs.rho_init = 250.0;  // Original stable rho
  tiny_InitWorkspace(&work, &info, &model, &data, &soln, &stgs);
  
  // Fill in the remaining struct (pass 0 for state constraints - not used)
  tiny_InitWorkspaceTemp(&work, &Qu, ZU, ZU_new, 0, 0);
  tiny_InitPrimalCache(&work, &Quu_inv, &AmBKt, &coeff_d2p);
  tiny_InitSolution(&work, Xhrz, Uhrz, 0, YU, 0, &Kinf, d, &Pinf, p);

  tiny_SetInitialState(&work, &x0);  
  tiny_SetStateReference(&work, Xref);
  tiny_SetInputReference(&work, Uref);
  // tiny_SetGoalState(&work, Xref, &xg);
  // tiny_SetGoalInput(&work, Uref, &ug);

  /* Set up LQR cost */
  tiny_InitDataCost(&work, &Q, q, &R, r, r_tilde);
  // R = R + stgs.rho_init * MatrixMf::Identity();
  // /* Set up constraints */
  ucu << 1 - u_hover[0], 1 - u_hover[1], 1 - u_hover[2], 1 - u_hover[3];
  lcu << -u_hover[0], -u_hover[1], -u_hover[2], -u_hover[3];
  tiny_SetInputBound(&work, &Acu, &lcu, &ucu);

  tiny_UpdateLinearCost(&work);

  /* Solver settings */
  stgs.en_cstr_goal = 0;
  stgs.en_cstr_inputs = 1;
  stgs.en_cstr_states = 0;  // No state constraints for basic test
  stgs.max_iter = benchIterSweep[0];
  stgs.verbose = 0;
  stgs.check_termination = 0;
  stgs.tol_abs_dual = 5e-2;
  stgs.tol_abs_prim = 5e-2;

  Klqr << 
  -0.123589f,0.123635f,0.285625f,-0.394876f,-0.419547f,-0.474536f,-0.073759f,0.072612f,0.186504f,-0.031569f,-0.038547f,-0.187738f,
  0.120236f,0.119379f,0.285625f,-0.346222f,0.403763f,0.475821f,0.071330f,0.068348f,0.186504f,-0.020972f,0.037152f,0.187009f,
  0.121600f,-0.122839f,0.285625f,0.362241f,0.337953f,-0.478858f,0.069310f,-0.070833f,0.186504f,0.022379f,0.015573f,-0.185212f,
  -0.118248f,-0.120176f,0.285625f,0.378857f,-0.322169f,0.477573f,-0.066881f,-0.070128f,0.186504f,0.030162f,-0.014177f,0.185941f;

  /* End of MPC initialization */  
  step = 0;  
  traj_iter = 0;

#if TINYMPC_BENCH_ONLY
  benchRequested = true;
  benchDone = false;
  benchHPoint = 0;
  benchIPoint = 0;
  benchIterInPoint = 0;
  for (int h = 0; h < BENCH_H_SWEEP_POINTS; ++h) {
    for (int p = 0; p < BENCH_I_SWEEP_POINTS; ++p) {
      benchMinUs[h][p] = 0xFFFFFFFFU;
      benchMaxUs[h][p] = 0U;
      benchSumUs[h][p] = 0U;
      benchMisses[h][p] = 0U;
      benchSolved[h][p] = 0U;
      benchMaxIterReached[h][p] = 0U;
      benchNonCvx[h][p] = 0U;
      benchOtherStatus[h][p] = 0U;
    }
  }
  DEBUG_PRINT("BENCH_ONLY mode active: motors forced off in controllerOutOfTree()\n");
  DEBUG_PRINT("CSV header: BENCH_CSV,H,iter,min_us,avg_us,max_us,misses,total,solved,max_iter,noncvx,other\n");
  DEBUG_PRINT("CSV header: BENCH_FIT_CSV,H,T0_us,k_iter_us_per_iter,k_step_iter_us_per_iter_step\n");
#else
  DEBUG_PRINT("Straight line trajectory (1m forward)\n");
#endif
}

bool controllerOutOfTreeTest() {
  // Always return true
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
#if TINYMPC_BENCH_ONLY
  (void)setpoint;
  (void)sensors;
  (void)state;
  (void)tick;
  control->normalizedForces[0] = 0.0f;
  control->normalizedForces[1] = 0.0f;
  control->normalizedForces[2] = 0.0f;
  control->normalizedForces[3] = 0.0f;
  control->controlMode = controlModePWM;
  return;
#endif

  // Get current time
  startTimestamp = usecTimestamp();

  /* Get current state (initial state for MPC) */
  // delta_x = x - x_bar; x_bar = 0
  // Positon error, [m]
  updateInitialState(sensors, state);

  /* Controller rate */
  if (RATE_DO_EXECUTE(MPC_RATE, tick)) { 
    // Get command reference
    updateHorizonReference(setpoint);

    /* MPC solve */
    // Solve optimization problem using ADMM
    tiny_UpdateLinearCost(&work);
    tiny_SolveAdmm(&work);
 
    result =  info.status_val * info.iter;
    
    // Detailed logging every 0.5 seconds
    static uint32_t mpc_log_counter = 0;
    if (mpc_log_counter % 50 == 0) {  // 100Hz / 50 = every 0.5s
      DEBUG_PRINT("MPC: pos=(%.2f,%.2f,%.2f) ref=(%.2f,%.2f,%.2f)\n", 
                  (double)x0(0), (double)x0(1), (double)x0(2),
                  (double)Xref[0](0), (double)Xref[0](1), (double)Xref[0](2));
      DEBUG_PRINT("MPC: u=(%.2f,%.2f,%.2f,%.2f) iter=%d\n",
                  (double)(Uhrz[0](0) + u_hover[0]), (double)(Uhrz[0](1) + u_hover[1]),
                  (double)(Uhrz[0](2) + u_hover[2]), (double)(Uhrz[0](3) + u_hover[3]),
                  info.iter);
    }
    mpc_log_counter++;
    
    // Position logging disabled
    // static uint32_t pos_log_counter = 0;
    // if (pos_log_counter % 50 == 0) {
    //   DEBUG_PRINT("POS: x=%.2f y=%.2f z=%.2f cstr=%d\n", 
    //               (double)x0(0), (double)x0(1), (double)x0(2), obs_constraint_active);
    // }
    // pos_log_counter++;
  }

  /* Output control — pure MPC, apply projected ADMM solution directly */
  if (setpoint->mode.z == modeDisable) {
    control->normalizedForces[0] = 0.0f;
    control->normalizedForces[1] = 0.0f;
    control->normalizedForces[2] = 0.0f;
    control->normalizedForces[3] = 0.0f;
  } else {
    control->normalizedForces[0] = ZU_new[0](0) + u_hover[0];  // PWM 0..1
    control->normalizedForces[1] = ZU_new[0](1) + u_hover[1];
    control->normalizedForces[2] = ZU_new[0](2) + u_hover[2];
    control->normalizedForces[3] = ZU_new[0](3) + u_hover[3];
  }
  control->controlMode = controlModePWM;
  // DEBUG_PRINT("pwm = [%.2f, %.2f]\n", (double)(control->normalizedForces[0]), (double)(control->normalizedForces[1]));

  // control->normalizedForces[0] = 0.0f;
  // control->normalizedForces[1] = 0.0f;
  // control->normalizedForces[2] = 0.0f;
  // control->normalizedForces[3] = 0.0f;
}

/**
 * Tunning variables for the full state quaternion LQR controller
 */
// PARAM_GROUP_START(ctrlMPC)
// /**
//  * @brief K gain
//  */
// PARAM_ADD(PARAM_FLOAT, u_hover, &u_hover)

// PARAM_GROUP_STOP(ctrlMPC)

/**
 * Logging variables for the command and reference signals for the
 * MPC controller
 */

// Note: LOG macros disabled due to C++ string literal compatibility with new firmware
/*
LOG_GROUP_START(ctrlMPC)

LOG_ADD(LOG_INT8, result, &result)
LOG_ADD(LOG_UINT32, mpcTime, &mpcTime)

LOG_ADD(LOG_FLOAT, u0, &(Uhrz[0](0)))
LOG_ADD(LOG_FLOAT, u1, &(Uhrz[0](1)))
LOG_ADD(LOG_FLOAT, u2, &(Uhrz[0](2)))
LOG_ADD(LOG_FLOAT, u3, &(Uhrz[0](3)))

LOG_ADD(LOG_FLOAT, zu0, &(ZU_new[0](0)))
LOG_ADD(LOG_FLOAT, zu1, &(ZU_new[0](1)))
LOG_ADD(LOG_FLOAT, zu2, &(ZU_new[0](2)))
LOG_ADD(LOG_FLOAT, zu3, &(ZU_new[0](3)))

LOG_GROUP_STOP(ctrlMPC)
*/

#ifdef __cplusplus
}
#endif
