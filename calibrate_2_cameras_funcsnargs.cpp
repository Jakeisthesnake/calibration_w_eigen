#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <tuple>
#include <string>
#include <sstream>
#include <cmath>
#include <utility>
#include <GL/glew.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>  // for AngleAxisRotatePoint, if you need to apply rotation
#include <ceres/autodiff_cost_function.h>  // for AutoDiffCostFunction (used in your case)
#include <ceres/solver.h>                  // for Solver options and summary
#include <ceres/problem.h>      

// loadAprilTagBoardFlat
using json = nlohmann::json;

// computeHomographies
using Point2dVec = std::vector<Eigen::Vector2d>;
using Point3dVec = std::vector<Eigen::Vector3d>;
using HomographyList = std::vector<Eigen::Matrix3d>;

// filterDataByTimestamps
// using Point2dVec = std::vector<Eigen::Vector2d>;
// using Point3dVec = std::vector<Eigen::Vector3d>;
using IDVec = std::vector<int>;
// using TimestampList = std::vector<double>;  // Or another type if needed

//Process CSV()
using Point3dVec = std::vector<Eigen::Vector3d>;
using Point2dVec = std::vector<Eigen::Vector2d>;
using IDVec = std::vector<int>;
using TimestampList = std::vector<uint64_t>;


std::unordered_map<int, Eigen::Vector3d> loadAprilTagBoardFlat(const std::string& json_file) {
    return id_to_point;
}

std::tuple<HomographyList, TimestampList> computeHomographies(
    const std::vector<Point3dVec>& obj_pts_list,
    const std::vector<Point2dVec>& img_pts_list,
    const TimestampList& timestamp_list)
{
    return {homographies, filtered_timestamps};
}





std::tuple<
    std::vector<Point3dVec>,
    std::vector<Point2dVec>,
    std::vector<IDVec>
> filterDataByTimestamps(
    const std::vector<Point3dVec>& obj_pts_list,
    const std::vector<Point2dVec>& img_pts_list,
    const std::vector<IDVec>& corner_ids_list,
    const TimestampList& timestamp_list,
    const TimestampList& filtered_timestamps)
{
    return {filtered_obj_pts, filtered_img_pts, filtered_corner_ids};
}


// Custom CSV reader or use something like io::CSVReader
// Dummy implementation here — you might replace it with a CSV library.

struct RowData {
    int cam_id;
    uint64_t timestamp_ns;
    double corner_x;
    double corner_y;
    int corner_id;
};

std::vector<RowData> readCSV(const std::string& file_path) {
    return rows;
}

Eigen::Vector3d get_object_point(
    int corner_id,
    int tag_rows = 6,
    int tag_cols = 6,
    double tag_size = 0.13,
    double tag_spacing = 0.04)
{
    return Eigen::Vector3d(tag_x + offset_x, tag_y + offset_y, 0.0);
}


std::tuple<
    std::vector<Point3dVec>,
    std::vector<Point2dVec>,
    std::vector<IDVec>,
    TimestampList
> processCSV(const std::string& file_path, int target_cam_id)
{
    return {obj_pts_list, img_pts_list, corner_ids_list, timestamp_list};
}


Eigen::Matrix3d compute_intrinsic_params(const std::vector<Eigen::Matrix3d>& H_list)
{
    return K;
}


std::pair<Eigen::Matrix3d, Eigen::Vector3d> compute_extrinsic_params(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K)
{
    return {R, t};
}



Eigen::MatrixXd kannala_brandt_project(
    const Eigen::MatrixXd& points,       // Nx3
    const Eigen::Vector4d& K,            // fx, fy, cx, cy
    const Eigen::Vector4d& dist_coeffs)  // k1, k2, k3, k4
{
    return projected;
}




// Skew-symmetric matrix
Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d w_hat;
    w_hat <<     0, -w(2),  w(1),
              w(2),     0, -w(0),
             -w(1),  w(0),     0;
    return w_hat;
}

// Inverse of the left Jacobian of SO(3)
Eigen::Matrix3d leftJacobianInverse(const Eigen::Vector3d& omega) {
    return Eigen::Matrix3d::Identity() - A * omega_hat + B * omega_hat_sq;
}

// Logarithm map of an SE(3) transformation matrix
Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d& T) {
    return result;
}

void visualize_camera_data(
    const std::vector<Eigen::Vector3d>& obj_pts_list_0,
    const std::vector<Eigen::Vector2d>& img_pts_list_0,
    const std::vector<Eigen::Vector2d>& projected_pts_0,
    const std::vector<Eigen::Vector3d>& obj_pts_list_1,
    const std::vector<Eigen::Vector2d>& img_pts_list_1,
    const std::vector<Eigen::Vector2d>& projected_pts_1)
{
}

Eigen::VectorXd fisheye_reprojection_error(
    const Eigen::VectorXd& params,
    const std::vector<Eigen::MatrixXd>& obj_pts_list_0,
    const std::vector<Eigen::MatrixXd>& img_pts_list_0,
    const std::vector<int>& timestamp_list_0,
    const std::vector<std::vector<int>>& corner_ids_list_0,
    const std::vector<Eigen::MatrixXd>& obj_pts_list_1,
    const std::vector<Eigen::MatrixXd>& img_pts_list_1,
    const std::vector<int>& timestamp_list_1,
    const std::vector<std::vector<int>>& corner_ids_list_1,
    const std::vector<int>& all_timestamps
) {
    Eigen::VectorXd result(total_error.size());
    for (size_t i = 0; i < total_error.size(); ++i)
        result(i) = total_error[i];

    std::cout << "total_error = " << result.sum() << std::endl;
    return result;
}


struct KannalaBrandtProjection {
    KannalaBrandtProjection(const Eigen::Vector3d& point_3d, const Eigen::Vector2d& observed)
        : point_3d_(point_3d), observed_(observed) {}
  
    template <typename T>
    bool operator()(const T* const K,           // fx, fy, cx, cy
                    const T* const dist_coeffs, // k1, k2, k3, k4
                    T* residuals) const {
      return true;
    }
  
    static ceres::CostFunction* Create(const Eigen::Vector3d& point_3d,
                                       const Eigen::Vector2d& observed) {
      return new ceres::AutoDiffCostFunction<KannalaBrandtProjection, 2, 4, 4>(
          new KannalaBrandtProjection(point_3d, observed));
    }
    
    private:
    const Eigen::Vector3d point_3d_;
    const Eigen::Vector2d observed_;
};

template<typename T>
Eigen::Matrix<T, 2, 1> projectKB(
    const Eigen::Matrix<T, 3, 1>& pt,
    const Eigen::Matrix<T, 3, 1>& rvec,
    const Eigen::Matrix<T, 3, 1>& tvec,
    const T* intrinsic,
    const T* dist_coeffs)
{
    return Eigen::Matrix<T, 2, 1>(u, v);
}


struct FisheyeReprojectionError {
    FisheyeReprojectionError(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt)
        : observed_pt_(observed_pt), obj_pt_(obj_pt) {}

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const distortion,
                    const T* const rvec, const T* const tvec,
                    T* residuals) const {
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt) {
        return (new ceres::AutoDiffCostFunction<FisheyeReprojectionError, 2, 4, 4, 3, 3>(
            new FisheyeReprojectionError(observed_pt, obj_pt)));
    }

    Eigen::Vector2d observed_pt_;
    Eigen::Vector3d obj_pt_;
};

void OptimizeFishEyeParameters(
    double intrinsic_0[4], double dist_0[4],
    std::vector<std::array<double, 6>>& extrinsics_0,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0,
    double intrinsic_1[4], double dist_1[4],
    std::vector<std::array<double, 6>>& extrinsics_1,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1,
    double rvec_cam_1[3], double tvec_cam_1[3])
{
    std::cout << summary.BriefReport() << std::endl;
}  
