#include "so3.hpp"
#include <cmath>

namespace so3 {
    Eigen::Matrix3d hat(const Eigen::Vector3d& omega) {
        Eigen::Matrix3d Omega;
        Omega << 0, -omega(2), omega(1),
                 omega(2), 0, -omega(0),
                 -omega(1), omega(0), 0;
        return Omega;
    }

    Eigen::Vector3d vee(const Eigen::Matrix3d& Omega) {
        Eigen::Vector3d omega;
        omega << Omega(2, 1), Omega(0, 2), Omega(1, 0);
        return omega;
    }

    Eigen::Matrix3d Exp(const Eigen::Vector3d& phi) {

        //rodrigues formula (Forster Eq 3)
        //use Taylor expansion for small angles. 1e-8 
        double theta = phi.norm();
        Eigen::Matrix3d Omega = hat(phi);
        if (theta < 1e-8) {
            return Eigen::Matrix3d::Identity() + Omega + 0.5 * (Omega * Omega); //2nd order taylor expansion.
        } else {
            return Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * Omega + ((1 - std::cos(theta)) / (theta * theta)) * (Omega * Omega);
        }

    }
    Eigen::Vector3d Log(const Eigen::Matrix3d& R) {                                                                                                                           
 
    
        double cos_theta = (R.trace() - 1) /2;
        double theta = std::acos(std::min(std::max(cos_theta, -1.0), 1.0)); // Clamp to [-1, 1]
        if (theta < 1e-8) {
            return vee(R - R.transpose()) / 2; // First-order approximation
        } else if (std::abs(theta - M_PI) < 1e-8) {
            // θ → π case: need alternative branch
            //Find the diagonal elements to give the length and the offdiagonal elements to give the direction.
            Eigen::Vector3d phi;
            phi(0) = std::sqrt((R(0, 0) + 1) / 2);
            phi(1) = std::sqrt((R(1, 1) + 1) / 2);
            phi(2) = std::sqrt((R(2, 2) + 1) / 2);
            // Determine the signs
            //choose the element with largest abs value. This is done to reduce errors from sensor noise.
            int max_index;
            phi.cwiseAbs().maxCoeff(&max_index);
            if (max_index == 0) {
                phi(1) = std::copysign(phi(1), R(0, 1));
                phi(2) = std::copysign(phi(2), R(0, 2));
            } else if (max_index == 1) {
                phi(0) = std::copysign(phi(0), R(1, 0));
                phi(2) = std::copysign(phi(2), R(1, 2));
            } else {
                phi(0) = std::copysign(phi(0), R(2, 0));
                phi(1) = std::copysign(phi(1), R(2, 1));
            }
            return theta * phi; // Scale by θ
        } else {
            return (theta / (2 * std::sin(theta))) * vee(R - R.transpose());
        }

    }  

    Eigen::Matrix3d Jr(const Eigen::Vector3d& phi) {
        double theta = phi.norm();
        Eigen::Matrix3d Omega = hat(phi);
        if (theta < 1e-8) {
            return Eigen::Matrix3d::Identity() - 0.5 * Omega + (1.0 / 6.0) * (Omega * Omega); // Taylor expansion
        } else {
            return Eigen::Matrix3d::Identity() - ((1 - std::cos(theta)) / (theta * theta)) * Omega + ((theta - std::sin(theta)) / (theta * theta * theta)) * (Omega * Omega);
        }
    }   


    Eigen::Matrix3d Jr_inv(const Eigen::Vector3d& phi) {
        // Compute the inverse of the right Jacobian for SO(3).
        // For small angles, Jr⁻¹(φ) ≈ I. For larger angles, use the exact formula:
        //   Jr⁻¹(φ) = I + 0.5 * hat(φ) + (1/φ² - (1 + cos(φ)) / (2 * φ * sin(φ))) * hat(φ)²
        double angle = phi.norm();
        if (angle < 1e-5) {
            Eigen::Matrix3d Phi = so3::hat(phi);
            return Eigen::Matrix3d::Identity()+ 0.5 * Phi + (1.0/12.0) * Phi * Phi; // Approximation for small angles
        } else {
            Eigen::Matrix3d hat_phi = so3::hat(phi);
            double s = sin(angle);
            double c = cos(angle);
            double factor = (1.0 / (angle * angle)) - ((1.0 + c) / (2.0 * angle * s));
            return Eigen::Matrix3d::Identity() + 0.5 * hat_phi + factor * hat_phi * hat_phi;
        }
    }


}