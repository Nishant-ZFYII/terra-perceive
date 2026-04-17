#include "imu_preintegration.hpp"
#include "so3.hpp"
#include <fstream>
#include <iostream>
#include <vector>
namespace imu {

    //everything that needs to be coded is in "imu_preintegration.hpp"
    Bias estimateBiasFromStatic(const std::vector<IMUMeasurement>& imu_data,int num_samples){

        Bias bias;
        for(int it = 0; it < num_samples; it++){
            bias.gyro += imu_data[it].omega;
            bias.accel += imu_data[it].accel;
        }

        bias.gyro /= num_samples;
        bias.accel /= num_samples;
        bias.accel -= kGravityWorld;

        return bias;

    }

    PreIntegrator::PreIntegrator(const Bias& bias) : bias_(bias) {
        reset();
        const double sg = 1.5e-3; //gyro noise
        const double sa = 6.0e-2; //accel noise
        Q_noise_ = Eigen::Matrix<double, 6, 6>::Zero();
        Q_noise_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * sg * sg;
        Q_noise_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * sa * sa;

    }

    void PreIntegrator::reset() {
        dR_ = Eigen::Matrix3d::Identity();
        dv_ = Eigen::Vector3d::Zero();
        dp_ = Eigen::Vector3d::Zero();
        dt_total_ = 0.0;
        Sigma_ = Eigen::Matrix<double, 9, 9>::Zero();

       
    }

    void PreIntegrator::integrate(const Eigen::Vector3d& omega,
                                  const Eigen::Vector3d& accel,
                                  double dt) {
    // Fold one IMU measurement into the accumulator.
    //   omega, accel: raw IMU readings (bias NOT yet subtracted)
    //   dt:           time since previous sample, seconds
    // Updates dR_, dv_, dp_, and Sigma_ in place.
    // YOUR CODE: bias-correct, update dp (first), dv, dR (last)
    //            Covariance propagation comes later
    
        Eigen::Vector3d omega_unbiased = omega - bias_.gyro;
        Eigen::Vector3d accel_unbiased = accel - bias_.accel;
        Eigen::Matrix3d dR_prev = dR_;
        // Update dp, dv, dR
        dp_ += dv_ * dt + 0.5 * (dR_ * accel_unbiased) * dt * dt;
        dv_ += (dR_ * accel_unbiased) * dt;
        dR_ = dR_ * so3::Exp(omega_unbiased * dt);
        // Update total time
        dt_total_ += dt;

        //covarience propogation (Forster Eq 35)
        //Σ_{k+1} = A · Σ_k · Aᵀ + B · Q · Bᵀ
        Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
        /*   A = [  ΔR_{k+1}ᵀ              0       0  ]     ← 3×3 blocks, total 9×9
                 [ -dR_k · hat(a_corr)·dt   I       0  ]
                 [ -0.5·dR_k·hat(a_corr)·dt²  I·dt  I  ]
        */
        A.block<3, 3>(0, 0) = so3::Exp(omega_unbiased * dt).transpose();
        A.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;
        A.block<3, 3>(3, 0) = -dR_prev * so3::hat(accel_unbiased) * dt;
        A.block<3, 3>(6, 0) = -0.5 * dR_prev * so3::hat(accel_unbiased) * dt * dt;
        Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
        //B
        /*  B = [ Jr(ω_corr · dt) · dt    0       ]     ← 9×6
                [     0                dR_k · dt   ]
                [     0            0.5·dR_k · dt²  ]*/
        B.block<3, 3>(0, 0) = so3::Jr(omega_unbiased * dt) * dt;
        B.block<3, 3>(3, 3) = dR_prev * dt;
        B.block<3, 3>(6, 3) = 0.5 * dR_prev * dt * dt;
        Sigma_ = A * Sigma_ * A.transpose() + B * Q_noise_ * B.transpose();

    }

    IMUFactor PreIntegrator::result(int frame_from, int frame_to) const {
        // Snapshot the current accumulated state as an IMUFactor.
        // YOUR CODE: construct and return the IMUFactor with current dR_, dv_, dp_, Sigma_.

        IMUFactor factor;
        factor.frame_from = frame_from;
        factor.frame_to = frame_to;
        factor.dt = dt_total_;
        factor.dR = dR_;
        factor.dv = dv_;
        factor.dp = dp_;
        factor.covariance = Sigma_;
        return factor;
    }

    std::vector<IMUFactor> preintegrateTrajectory(
                            const std::vector<IMUMeasurement>& imu_data,
                            const std::vector<double>& lidar_times,
                            const Bias& bias) {
        // Pre-integrate IMU measurements between keyframes.
        
        int N = lidar_times.size();
        std::vector<IMUFactor> factors(N - 1);
        PreIntegrator integrator(bias);
        int imu_idx = 0;

        for (int i = 0; i < N - 1; i++) {
            double t_start = lidar_times[i];
            double t_end = lidar_times[i + 1];
            double t_prev = t_start;
            integrator.reset();
            while (imu_idx < imu_data.size() && imu_data[imu_idx].t < t_end) {
                if (imu_data[imu_idx].t >= t_start) {
                    double dt = imu_data[imu_idx].t - t_prev;
                    integrator.integrate(imu_data[imu_idx].omega, imu_data[imu_idx].accel, dt);
                    t_prev = imu_data[imu_idx].t;
                }
                imu_idx++;
            }
            factors[i] = integrator.result(i, i + 1);
        }

        return factors;
    }

    std::vector<IMUMeasurement> loadIMUFromCSV(const std::string& path){
        //'data/imu_raw.csv'
        std::vector<IMUMeasurement> imu_data;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return imu_data;
        }
        std::string line;
        std::getline(file, line); // Skip header
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            IMUMeasurement meas;
            //comma seperated data, need to consume the comma
            char comma;
            ss >> meas.t >> comma >> meas.omega.x() >> comma >> meas.omega.y() >> comma >> meas.omega.z()
               >> comma >> meas.accel.x() >> comma >> meas.accel.y() >> comma >> meas.accel.z();
            imu_data.push_back(meas);
        }
        return imu_data;

    }

    Bias loadBiasFromCSV(const std::string& path){
        //'data/imu_bias.csv'
        Bias bias;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return bias;
        }
        std::string line;
        std::getline(file, line); // Skip header
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            char comma;
            ss >> bias.gyro.x() >> comma >> bias.gyro.y() >> comma >> bias.gyro.z()
               >> comma >> bias.accel.x() >> comma >> bias.accel.y() >> comma >> bias.accel.z();
        }
        return bias;
    }

    void saveFactorsToCSV(const std::vector<IMUFactor>& factors,
                      const std::string& path){

        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return;
        }
        // Write header
        file << "frame_from,frame_to,dt,"
             << "dR_00,dR_01,dR_02,dR_10,dR_11,dR_12,dR_20,dR_21,dR_22,"
             << "dv_x,dv_y,dv_z,"
             << "dp_x,dp_y,dp_z,"
             << "cov_00,cov_01,cov_02,cov_03,cov_04,cov_05,cov_06,cov_07,cov_08,"
             << "cov_10,cov_11,cov_12,cov_13,cov_14,cov_15,cov_16,cov_17,cov_18,"
             << "cov_20,cov_21,cov_22,cov_23,cov_24,cov_25,cov_26,cov_27,cov_28,"
             << "cov_30,cov_31,cov_32,cov_33,cov_34,cov_35,cov_36,cov_37,cov_38,"
             << "cov_40,cov_41,cov_42,cov_43,cov_44,cov_45,cov_46,cov_47,cov_48,"
             << "cov_50,cov_51,cov_52,cov_53,cov_54,cov_55,cov_56,cov_57,cov_58,"
             << "cov_60,cov_61,cov_62,cov_63,cov_64,cov_65,cov_66,cov_67,cov_68," 
             << "cov_70,cov_71,cov_72,cov_73,cov_74,cov_75,cov_76,cov_77,cov_78,"
             << "cov_80,cov_81,cov_82,cov_83,cov_84,cov_85,cov_86,cov_87,cov_88\n";
        
        // Write data
        for (const auto& factor : factors) {
            file << factor.frame_from << "," << factor.frame_to << "," << factor.dt << ",";
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    file << factor.dR(i, j) << ",";
                }
            }
            file << factor.dv.x() << "," << factor.dv.y() << "," << factor.dv.z() << ",";
            file << factor.dp.x() << "," << factor.dp.y() << "," << factor.dp.z() << ",";
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    file << factor.covariance(i, j);
                    if (i < 8 || j < 8) file << ",";
                }
            }
            file << "\n";
        }
        
    }
}  // namespace imu

