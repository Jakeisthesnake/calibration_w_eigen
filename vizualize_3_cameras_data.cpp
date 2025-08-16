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
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>  // for AngleAxisRotatePoint, if you need to apply rotation
#include <ceres/autodiff_cost_function.h>  // for AutoDiffCostFunction (used in your case)
#include <ceres/solver.h>                  // for Solver options and summary
#include <ceres/problem.h>    
#include <array>
#include <algorithm>
#include <random>
#include <iostream>
#include <pangolin/handler/handler.h>
#include <GL/freeglut.h>


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





struct CSVRow {
    uint64_t timestamp_ns;
    int camera_id;
    int corner_id;
    double x;
    double y;
    double radius;
};

std::vector<CSVRow> readCSV(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<CSVRow> rows;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return rows;
    }

    std::string line;
    bool is_first_line = true;

    while (std::getline(file, line)) {
        if (is_first_line) {
            is_first_line = false; // Skip header
            continue;
        }

        std::stringstream ss(line);
        std::string token;

        CSVRow row;

        std::getline(ss, token, ',');
        row.timestamp_ns = std::stoull(token);

        std::getline(ss, token, ',');
        row.camera_id = std::stoi(token);

        std::getline(ss, token, ',');
        row.corner_id = std::stoi(token);

        std::getline(ss, token, ',');
        row.x = std::stod(token);

        std::getline(ss, token, ',');
        row.y = std::stod(token);

        std::getline(ss, token, ',');
        row.radius = std::stod(token);

        rows.push_back(row);
    }

    return rows;
}

Eigen::Vector3d get_object_point(
    int corner_id,
    int tag_rows = 6,
    int tag_cols = 6,
    double tag_size = 0.13,
    double tag_spacing = 0.04)
{
    int tag_index = corner_id / 4;
    int local_corner = corner_id % 4;

    int row = tag_index / tag_cols;
    int col = tag_index % tag_cols;

    double tag_x = col * (tag_size + tag_spacing);
    double tag_y = row * (tag_size + tag_spacing);

    // Offsets for corners: TL, TR, BR, BL
    const double corner_offsets[4][2] = {
        {0.0, 0.0},                // Top-left
        {tag_size, 0.0},            // Top-right
        {tag_size, tag_size},        // Bottom-right
        {0.0, tag_size}             // Bottom-left
    };

    double offset_x = corner_offsets[local_corner][0];
    double offset_y = corner_offsets[local_corner][1];

    return Eigen::Vector3d(tag_x + offset_x, tag_y + offset_y, 0.0);
}

std::vector<Eigen::Vector3d> loadAprilTagBoardFlat(const std::string& json_path)
{
    std::ifstream file(json_path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + json_path);

    nlohmann::json j;
    file >> j;

    int tag_cols = j["tagCols"];
    int tag_rows = j["tagRows"];
    double tag_size = j["tagSize"];
    double tag_spacing = j["tagSpacing"];

    int total_tags = tag_cols * tag_rows;
    int total_corners = total_tags * 4;

    std::vector<Eigen::Vector3d> object_points;
    object_points.reserve(total_corners);

    for (int corner_id = 0; corner_id < total_corners; ++corner_id)
    {
        object_points.push_back(
            get_object_point(corner_id, tag_rows, tag_cols, tag_size, tag_spacing)
        );
    }

    return object_points;
}


std::tuple<
    std::vector<Point3dVec>,
    std::vector<Point2dVec>,
    std::vector<IDVec>,
    TimestampList
> processCSV(const std::string& file_path, int target_cam_id)
{
    auto rows = readCSV(file_path);

    // Grouped output per timestamp
    struct DataGroup {
        Point3dVec obj_points;
        Point2dVec img_points;
        IDVec corner_ids;
    };
    std::unordered_map<uint64_t, DataGroup> grouped_data;

    for (const auto& row : rows) {
        if (row.camera_id != target_cam_id) continue;

        Eigen::Vector2d img_pt(row.x, row.y);
        Eigen::Vector3d obj_pt = get_object_point(row.corner_id);

        auto& group = grouped_data[row.timestamp_ns];
        group.img_points.push_back(img_pt);
        group.obj_points.push_back(obj_pt);
        group.corner_ids.push_back(row.corner_id);
    }

    // Sort timestamps
    std::vector<uint64_t> sorted_timestamps;
    sorted_timestamps.reserve(grouped_data.size());
    for (const auto& [timestamp, _] : grouped_data) {
        sorted_timestamps.push_back(timestamp);
    }
    std::sort(sorted_timestamps.begin(), sorted_timestamps.end());

    // Extract data in sorted order
    std::vector<Point3dVec> obj_pts_list;
    std::vector<Point2dVec> img_pts_list;
    std::vector<IDVec> corner_ids_list;
    TimestampList timestamp_list;

    for (const auto& timestamp : sorted_timestamps) {
        const auto& data = grouped_data[timestamp];
        obj_pts_list.push_back(data.obj_points);
        img_pts_list.push_back(data.img_points);
        corner_ids_list.push_back(data.corner_ids);
        timestamp_list.push_back(timestamp);
    }

    return {obj_pts_list, img_pts_list, corner_ids_list, timestamp_list};
}


bool LoadCalibrationResult(
    const std::string& filepath,

    double intrinsic_0[4], double dist_0[4],
    std::vector<std::array<double, 7>>& target_poses,
    std::vector<double>& timestamps,

    double intrinsic_1[4], double dist_1[4],

    double intrinsic_2[4], double dist_2[4],

    double qvec_cam_1[4], double tvec_cam_1[3],
    double qvec_cam_2[4], double tvec_cam_2[3]
) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open calibration file: " << filepath << std::endl;
        return false;
    }

    json input;
    ifs >> input;

    auto load_array4 = [](const json& j, double arr[4]) {
        for (int i = 0; i < 4; ++i) arr[i] = j[i].get<double>();
    };
    auto load_array3 = [](const json& j, double arr[3]) {
        for (int i = 0; i < 3; ++i) arr[i] = j[i].get<double>();
    };

    // --- Intrinsics & Distortion ---
    load_array4(input["camera0"]["intrinsics"], intrinsic_0);
    load_array4(input["camera0"]["distortion"], dist_0);

    load_array4(input["camera1"]["intrinsics"], intrinsic_1);
    load_array4(input["camera1"]["distortion"], dist_1);

    load_array4(input["camera2"]["intrinsics"], intrinsic_2);
    load_array4(input["camera2"]["distortion"], dist_2);

    // --- Target poses in world frame ---
    target_poses.clear();
    timestamps.clear();
    for (const auto& pose : input["target_poses"]) {
        std::array<double, 7> tp{};
        // quaternion
        for (int i = 0; i < 4; ++i) tp[i] = pose["quaternion"][i].get<double>();
        // translation
        for (int i = 0; i < 3; ++i) tp[4 + i] = pose["translation"][i].get<double>();
        target_poses.push_back(tp);
        timestamps.push_back(pose["timestamp"].get<double>());
    }

    // --- Inter-camera transforms (quaternions + translations) ---
    load_array4(input["inter_camera"]["camera1_to_camera0"]["quaternion"], qvec_cam_1);
    load_array3(input["inter_camera"]["camera1_to_camera0"]["translation_vector"], tvec_cam_1);

    load_array4(input["inter_camera"]["camera2_to_camera0"]["quaternion"], qvec_cam_2);
    load_array3(input["inter_camera"]["camera2_to_camera0"]["translation_vector"], tvec_cam_2);

    return true;
}


Eigen::MatrixXd kannala_brandt_project(
    const Eigen::MatrixXd& points,       // Nx3
    const Eigen::Vector4d& K,            // fx, fy, cx, cy
    const Eigen::Vector4d& dist_coeffs)  // k1, k2, k3, k4
{
    const double k1 = dist_coeffs(0);
    const double k2 = dist_coeffs(1);
    const double k3 = dist_coeffs(2);
    const double k4 = dist_coeffs(3);

    const double fx = K(0);
    const double fy = K(1);
    const double cx = K(2);
    const double cy = K(3);

    const int N = points.rows();

    // Split coordinates
    Eigen::VectorXd X = points.col(0);
    Eigen::VectorXd Y = points.col(1);
    Eigen::VectorXd Z = points.col(2);

    Eigen::VectorXd r = (X.array().square() + Y.array().square()).sqrt();
    Eigen::VectorXd theta = (r.array() > 1e-8).select((r.array() / Z.array()).atan(), 0.0);

    Eigen::VectorXd theta2 = theta.array().square();
    Eigen::VectorXd theta4 = theta2.array().square();
    Eigen::VectorXd theta6 = theta2.array() * theta4.array();
    Eigen::VectorXd theta8 = theta4.array().square();

    Eigen::VectorXd theta_d = (theta.array()
        + k1 * theta2.array() * theta.array()
        + k2 * theta4.array() * theta.array()
        + k3 * theta6.array() * theta.array()
        + k4 * theta8.array() * theta.array()).matrix();

    Eigen::VectorXd scale = (r.array() > 1e-8).select(theta_d.array() / r.array(), 1.0);

    Eigen::VectorXd x_distorted = X.array() * scale.array();
    Eigen::VectorXd y_distorted = Y.array() * scale.array();

    Eigen::MatrixXd projected(N, 2);
    projected.col(0) = fx * x_distorted.array() + cx;
    projected.col(1) = fy * y_distorted.array() + cy;

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
    double theta = omega.norm();

    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix3d omega_hat = skew(omega);
    Eigen::Matrix3d omega_hat_sq = omega_hat * omega_hat;

    double A = 0.5;
    double B = (1.0 / (theta * theta)) -
              ((1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta)));

    return Eigen::Matrix3d::Identity() - A * omega_hat + B * omega_hat_sq;
}

// Logarithm map of an SE(3) transformation matrix
Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);

    double trace_R = R.trace();
    double cos_theta = std::min(std::max((trace_R - 1.0) / 2.0, -1.0), 1.0);
    double theta = std::acos(cos_theta);

    Eigen::Vector3d omega;
    Eigen::Matrix3d J_inv;

    if (theta < 1e-8) {
        omega.setZero();
        J_inv = Eigen::Matrix3d::Identity();
    } else {
        omega = (theta / (2.0 * std::sin(theta))) * Eigen::Vector3d(
            R(2,1) - R(1,2),
            R(0,2) - R(2,0),
            R(1,0) - R(0,1)
        );
        J_inv = leftJacobianInverse(omega);
    }

    Eigen::Vector3d upsilon = J_inv * t;

    Eigen::Matrix<double, 6, 1> result;
    result.head<3>() = omega;
    result.tail<3>() = upsilon;

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
    pangolin::CreateWindowAndBind("Camera Calibration Visualization", 1280, 720);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 100),
        pangolin::ModelViewLookAt(1, -2, 3, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 0.7, -1280.0/720.0)
                                .SetHandler(&handler);
    

    pangolin::View& d_2d = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.7, 1.0)
                                .SetLayout(pangolin::LayoutEqual);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw 3D view
        d_cam.Activate(s_cam);
        glPointSize(5.0);

        // Draw Camera 0 Object Points
        glColor3f(0.0, 0.0, 1.0);
        glBegin(GL_POINTS);
        for (const auto& pt : obj_pts_list_0)
            glVertex3d(pt[0], pt[1], pt[2]);
        glEnd();

        // Draw Camera 1 Object Points
        glColor3f(0.5, 0.5, 1.0);
        glBegin(GL_POINTS);
        for (const auto& pt : obj_pts_list_1)
            glVertex3d(pt[0], pt[1], pt[2]);
        glEnd();

        // 2D Viewport for image vs. projected
        d_2d.Activate();

        glPointSize(3.0);
        glBegin(GL_POINTS);
        // Camera 0 - Image Points (Red)
        glColor3f(1.0, 0.0, 0.0);
        for (const auto& pt : img_pts_list_0)
            glVertex2d(pt[0], pt[1]);

        // Camera 0 - Projected Points (Green)
        glColor3f(0.0, 1.0, 0.0);
        for (const auto& pt : projected_pts_0)
            glVertex2d(pt[0], pt[1]);

        // Camera 1 - Image Points (Orange)
        glColor3f(1.0, 0.5, 0.0);
        for (const auto& pt : img_pts_list_1)
            glVertex2d(pt[0], pt[1]);

        // Camera 1 - Projected Points (Cyan)
        glColor3f(0.0, 1.0, 1.0);
        for (const auto& pt : projected_pts_1)
            glVertex2d(pt[0], pt[1]);
        glEnd();

        pangolin::FinishFrame();
    }
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
    int num_images_0 = timestamp_list_0.size();
    int num_images_1 = timestamp_list_1.size();

    int cam_0_param_length = 8 + num_images_0 * 6;
    int cam_1_param_length = cam_0_param_length + 8 + num_images_1 * 6;

    // Parse parameters
    Eigen::Vector4d K_0 = params.segment<4>(0);
    Eigen::Vector4d dist_coeffs_0 = params.segment<4>(4);
    Eigen::MatrixXd extrinsics_0 = Eigen::Map<const Eigen::MatrixXd>(params.data() + 8, 6, num_images_0).transpose();

    Eigen::Vector4d K_1 = params.segment<4>(cam_0_param_length);
    Eigen::Vector4d dist_coeffs_1 = params.segment<4>(cam_0_param_length + 4);
    Eigen::MatrixXd extrinsics_1 = Eigen::Map<const Eigen::MatrixXd>(params.data() + cam_0_param_length + 8, 6, num_images_1).transpose();

    Eigen::Vector3d rvec_cam_1 = params.segment<3>(cam_1_param_length);
    Eigen::Vector3d tvec_cam_1 = params.segment<3>(cam_1_param_length + 3);
    Eigen::Matrix3d R_matrix_cam_1 = Eigen::AngleAxisd(rvec_cam_1.norm(), rvec_cam_1.normalized()).toRotationMatrix();

    std::vector<double> total_error;

    for (int i = 0; i < all_timestamps.size(); ++i) {
        int ts = all_timestamps[i];
        int cam_0_index = -1, cam_1_index = -1;
        for (int j = 0; j < timestamp_list_0.size(); ++j)
            if (timestamp_list_0[j] == ts) cam_0_index = j;
        for (int j = 0; j < timestamp_list_1.size(); ++j)
            if (timestamp_list_1[j] == ts) cam_1_index = j;

        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d t0, t1;

        if (cam_0_index != -1) {
            Eigen::Vector3d rvec = extrinsics_0.row(cam_0_index).head<3>();
            t0 = extrinsics_0.row(cam_0_index).tail<3>();
            R0 = Eigen::AngleAxisd(rvec.norm(), rvec.normalized()).toRotationMatrix();

            Eigen::MatrixXd obj_pts_3d = Eigen::MatrixXd::Zero(obj_pts_list_0[cam_0_index].rows(), 3);
            obj_pts_3d.leftCols(2) = obj_pts_list_0[cam_0_index];
            Eigen::MatrixXd transformed = (R0 * obj_pts_3d.transpose()).colwise() + t0;
            transformed.transposeInPlace();

            auto projected = kannala_brandt_project(transformed, K_0, dist_coeffs_0);
            Eigen::MatrixXd err = projected.cast<double>() - img_pts_list_0[cam_0_index].cast<double>();
            for (int j = 0; j < err.size(); ++j)
                total_error.push_back(err(j));
        }

        if (cam_1_index != -1) {
            Eigen::Vector3d rvec = extrinsics_1.row(cam_1_index).head<3>();
            t1 = extrinsics_1.row(cam_1_index).tail<3>();
            R1 = Eigen::AngleAxisd(rvec.norm(), rvec.normalized()).toRotationMatrix();

            Eigen::MatrixXd obj_pts_3d = Eigen::MatrixXd::Zero(obj_pts_list_1[cam_1_index].rows(), 3);
            obj_pts_3d.leftCols(2) = obj_pts_list_1[cam_1_index];
            Eigen::MatrixXd transformed = (R1 * obj_pts_3d.transpose()).colwise() + t1;
            transformed.transposeInPlace();

            auto projected = kannala_brandt_project(transformed, K_1, dist_coeffs_1);
            Eigen::MatrixXd err = projected.cast<double>() - img_pts_list_1[cam_1_index].cast<double>();
            for (int j = 0; j < err.size(); ++j)
                total_error.push_back(err(j));
        }

        if (cam_0_index != -1 && cam_1_index != -1) {
            Eigen::Matrix4d T_0 = Eigen::Matrix4d::Identity();
            T_0.topLeftCorner<3, 3>() = R0;
            T_0.topRightCorner<3, 1>() = t0;

            Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
            T_1.topLeftCorner<3, 3>() = R1;
            T_1.topRightCorner<3, 1>() = t1;

            Eigen::Matrix4d T_01_obs = Eigen::Matrix4d::Identity();
            T_01_obs.topLeftCorner<3, 3>() = R_matrix_cam_1;
            T_01_obs.topRightCorner<3, 1>() = tvec_cam_1;

            Eigen::Matrix4d T_01_est = T_0 * T_1.inverse();

            Eigen::VectorXd pose_error = logSE3(T_01_obs * T_01_est.inverse());
            for (int j = 0; j < pose_error.size(); ++j)
                total_error.push_back(pose_error(j));
        }
    }

    Eigen::VectorXd result(total_error.size());
    for (size_t i = 0; i < total_error.size(); ++i)
        result(i) = total_error[i];

    std::cout << "total_error = " << result.sum() << std::endl;
    return result;
}


struct TimestampEntry {
    size_t timestamp_id;
    int cam0_idx;  // -1 if missing
    int cam1_idx;  // -1 if missing
    int cam2_idx;  // -1 if missing
};


struct FrameData {
    std::vector<Eigen::Vector2d> observed_pts;
    std::vector<Eigen::Vector2d> projected_pts;
    std::vector<Eigen::Vector3d> object_pts;
    std::array<double, 6> extrinsics;
    double error_sum = 0.0;
    int frame_index = -1;
    uint64_t timestamp_ns = -1;
};


std::vector<FrameData> GenerateReprojectionErrorData(
    const double* intrinsic,
    const double* dist,
    const std::vector<std::array<double, 7>>& extrinsics,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts,
    const TimestampList& timestamps_ns)
{

    std::vector<FrameData> result;

    Eigen::Map<const Eigen::Vector4d> K(intrinsic);
    Eigen::Map<const Eigen::Vector4d> dist_coeffs(dist);
    std::cout << "Generate img_pts.size() = " << img_pts.size() << std::endl;

    if (timestamps_ns.size() != img_pts.size()) {
        throw std::runtime_error("Timestamps size does not match frame count.");
    }

    for (size_t i = 0; i < img_pts.size(); ++i) {
        FrameData frame;
        frame.timestamp_ns = timestamps_ns[i];  // <-- use uint64_t timestamp

        const auto& observed = img_pts[i];
        const auto& object = obj_pts[i];
        const auto& ext = extrinsics[i];

        Eigen::Quaterniond q(ext[0], ext[1], ext[2], ext[3]);
        Eigen::Vector3d tvec(ext[4], ext[5], ext[6]);

        const size_t N = observed.size();
        Eigen::MatrixXd points_cam(N, 3);

        for (size_t j = 0; j < N; ++j) {
            points_cam.row(j) = (q * object[j] + tvec).transpose();
        }

        Eigen::MatrixXd projected = kannala_brandt_project(points_cam, K, dist_coeffs);

        double error_sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            frame.observed_pts.push_back(observed[j]);
            frame.projected_pts.push_back(projected.row(j).transpose());
            error_sum += (observed[j] - projected.row(j).transpose()).norm();
        }

        frame.error_sum = error_sum;
        frame.frame_index = static_cast<int>(i);
        result.push_back(std::move(frame));
        // std::cout << "Frame " << i << ": timestamp = " << frame.timestamp_ns
        //           << ", error_sum = " << frame.error_sum << std::endl;
        // std::cout << "Extrinsics (qvec + tvec): " 
        //           << ext[0] << ", " << ext[1] << ", " << ext[2] << ", " << ext[3] << ", "
        //           << ext[4] << ", " << ext[5] << ", " << ext[6] << std::endl;
        // std::cout << "Observed points: ";
        // for (const auto& pt : observed) {
        //     std::cout << "(" << pt.x() << ", " << pt.y() << ") ";
        // }
        // std::cout << std::endl;
        // std::cout << "Projected points: ";
        // for (int i = 0; i < projected.rows(); ++i) {
        //     Eigen::Vector2d pt = projected.block<1, 2>(i, 0);
        //     std::cout << "(" << pt.x() << ", " << pt.y() << ") ";
        // }
        
        // std::cout << "Intrinsics: "
        //           << intrinsic[0] << ", " << intrinsic[1] << ", "
        //           << intrinsic[2] << ", " << intrinsic[3] << std::endl;
        // std::cout << "Distortion: "
        //           << dist[0] << ", " << dist[1] << ", "
        //           << dist[2] << ", " << dist[3] << std::endl;
        // std::cout << "----------------------------------------" << std::endl;
    }

    std::cout << "Generated " << result.size() << " frames of reprojection error data." << std::endl;
    return result;
}


using FrameMap = std::unordered_map<int64_t, FrameData>;

// --------------------------- Types ----------------------------
struct FrameDetections {
    // Observed 2D points in pixel coordinates for this frame (order matches board_points)
    std::vector<Eigen::Vector2d> observations;
    int64_t timestamp_ns = 0;
};

struct CameraInit {
    // intrinsics: fx, fy, cx, cy
    double K[4]  = {600,600,640,480};
    // fisheye KB (equidistant) k1..k4
    double D[4]  = {0,0,0,0};
};

struct InterCamInit {
    // camX -> cam0
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero(); // roll,pitch,yaw (rad)
    Eigen::Vector3d t   = Eigen::Vector3d::Zero(); // meters
};

struct BoardPoseInit {
    // Per frame initial target->cam0 pose
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
    Eigen::Vector3d t   = Eigen::Vector3d(0,0,1.0);
};

// ---------------------- Math helpers -------------------------
inline Eigen::Matrix3d R_from_rpy(const Eigen::Vector3d& rpy) {
    // rpy = (roll, pitch, yaw)
    double cr = std::cos(rpy.x()), sr = std::sin(rpy.x());
    double cp = std::cos(rpy.y()), sp = std::sin(rpy.y());
    double cy = std::cos(rpy.z()), sy = std::sin(rpy.z());
    // ZYX yaw-pitch-roll
    Eigen::Matrix3d R;
    R << cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
         sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
         -sp,        cp*sr,              cp*cr;
    return R;
}

// Kannala–Brandt (equidistant) fisheye projection with 4 coefficients (k1..k4)
inline bool ProjectFisheyeKB(
    const Eigen::Vector3d& Pc,          // point in camera frame
    const double K[4],                  // fx,fy,cx,cy
    const double D[4],                  // k1..k4
    Eigen::Vector2d& uv                 // output pixel
) {
    const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
    if (Z <= 1e-9) return false; // behind camera or too close to plane

    const double r = std::sqrt(X*X + Y*Y);
    const double theta = std::atan2(r, Z);
    double theta2 = theta*theta;
    double theta4 = theta2*theta2;
    double theta6 = theta4*theta2;
    double theta8 = theta4*theta4;

    const double k1 = D[0], k2 = D[1], k3 = D[2], k4 = D[3];
    const double theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8);

    double scale = (r > 1e-12) ? (theta_d / r) : 0.0;
    double xd = X * scale;
    double yd = Y * scale;

    const double fx = K[0], fy = K[1], cx = K[2], cy = K[3];
    uv.x() = fx * xd + cx;
    uv.y() = fy * yd + cy;
    return std::isfinite(uv.x()) && std::isfinite(uv.y());
}

// Transform points: X_cam0 = R0 * X_target + t0
inline Eigen::Vector3d ToCam0(
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& t0,
    const Eigen::Vector3d& X_target
) {
    return R0 * X_target + t0;
}

// For camX with camX->cam0 known: X_cam0 = R_x0 * X_camX + t_x0
// => X_camX = R_x0^T * (X_cam0 - t_x0)
inline Eigen::Vector3d ToCamX_fromCam0(
    const Eigen::Matrix3d& R_x0,
    const Eigen::Vector3d& t_x0,
    const Eigen::Vector3d& X_cam0
) {
    return R_x0.transpose() * (X_cam0 - t_x0);
}

// RMS error
inline double ComputeRMSError(
    const std::vector<Eigen::Vector2d>& obs,
    const std::vector<Eigen::Vector2d>& proj
) {
    if (obs.size() == 0 || proj.size() != obs.size()) return 0.0;
    double sum2 = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < obs.size(); ++i) {
        if (!std::isfinite(proj[i].x()) || !std::isfinite(proj[i].y())) continue;
        Eigen::Vector2d d = obs[i] - proj[i];
        sum2 += d.squaredNorm();
        ++n;
    }
    return (n > 0) ? std::sqrt(sum2 / n) : 0.0;
}

// ---------------------- Visualization ------------------------
void VisualizeStereoReprojectionTuner(
    // AprilTag board geometry (object points in board/target frame; order must match detections)
    const std::vector<Eigen::Vector3d>& board_points,

    // Per-frame detections for each camera (same indexing across cams; empty vectors allowed when not detected)
    const std::vector<FrameDetections>& frames_cam0,
    const std::vector<FrameDetections>& frames_cam1,
    const std::vector<FrameDetections>& frames_cam2,

    // Initial intrinsics & fisheye params for each camera
    const CameraInit& cam0_init,
    const CameraInit& cam1_init,
    const CameraInit& cam2_init,

    // Initial inter-camera transforms (cam1->cam0, cam2->cam0)
    const InterCamInit& cam1_to_cam0_init,
    const InterCamInit& cam2_to_cam0_init,

    // Initial board poses target->cam0 per frame (same size as frames)
    const std::vector<BoardPoseInit>& board_poses_init
) {
    // Basic checks (soft)
    const size_t N = std::max(frames_cam0.size(), std::max(frames_cam1.size(), frames_cam2.size()));
    if (board_poses_init.size() != N) {
        std::cerr << "Warning: board_poses_init size != number of frames; clamping to min.\n";
    }

    // Build union of timestamps for stepping (optional; we’ll just use index)
    std::vector<int64_t> timestamps;
    timestamps.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        int64_t ts = 0;
        if (i < frames_cam0.size()) ts = std::max<int64_t>(ts, frames_cam0[i].timestamp_ns);
        if (i < frames_cam1.size()) ts = std::max<int64_t>(ts, frames_cam1[i].timestamp_ns);
        if (i < frames_cam2.size()) ts = std::max<int64_t>(ts, frames_cam2[i].timestamp_ns);
        timestamps.push_back(ts);
    }

    // ---------------- Pangolin setup ----------------
    pangolin::CreateWindowAndBind("Stereo Fisheye Reprojection Tuner", 1700, 900);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // A simple 2D canvas that spans 3 images in a row
    // We'll derive a nominal image size from cam0 cx, cy
    const int img_w0 = (int)std::round(2.0 * cam0_init.K[2]);
    const int img_h0 = (int)std::round(2.0 * cam0_init.K[3]);
    const int panel_w = std::max(img_w0, 640);
    const int panel_h = std::max(img_h0, 480);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrixOrthographic(0, 3.0*panel_w, panel_h, 0, -1, 1)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(300), 1.0) // leave space on the left for UI
        .SetAspect( (float)(3.0*panel_w) / (float)panel_h )
        .SetHandler(&pangolin::StaticHandler);

    // ---------------- UI Sliders ----------------
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(300));

    // Frame control
    pangolin::Var<int> frame_idx("ui.frame", 0, 0, (int)std::max<int>(0, (int)N-1));
    pangolin::Var<bool> prev_f("ui.prev_frame", false, false);
    pangolin::Var<bool> next_f("ui.next_frame", false, false);

    // Cam0 intrinsics & distortion
    pangolin::Var<double> fx0("ui.fx0", cam0_init.K[0], 50, 5000);
    pangolin::Var<double> fy0("ui.fy0", cam0_init.K[1], 50, 5000);
    pangolin::Var<double> cx0("ui.cx0", cam0_init.K[2], 0, 4000);
    pangolin::Var<double> cy0("ui.cy0", cam0_init.K[3], 0, 4000);
    pangolin::Var<double> k0_1("ui.k0_1", cam0_init.D[0], -2, 2);
    pangolin::Var<double> k0_2("ui.k0_2", cam0_init.D[1], -2, 2);
    pangolin::Var<double> k0_3("ui.k0_3", cam0_init.D[2], -2, 2);
    pangolin::Var<double> k0_4("ui.k0_4", cam0_init.D[3], -2, 2);

    // Cam1 intrinsics & distortion
    pangolin::Var<double> fx1("ui.fx1", cam1_init.K[0], 50, 5000);
    pangolin::Var<double> fy1("ui.fy1", cam1_init.K[1], 50, 5000);
    pangolin::Var<double> cx1("ui.cx1", cam1_init.K[2], 0, 4000);
    pangolin::Var<double> cy1("ui.cy1", cam1_init.K[3], 0, 4000);
    pangolin::Var<double> k1_1("ui.k1_1", cam1_init.D[0], -2, 2);
    pangolin::Var<double> k1_2("ui.k1_2", cam1_init.D[1], -2, 2);
    pangolin::Var<double> k1_3("ui.k1_3", cam1_init.D[2], -2, 2);
    pangolin::Var<double> k1_4("ui.k1_4", cam1_init.D[3], -2, 2);

    // Cam2 intrinsics & distortion
    pangolin::Var<double> fx2("ui.fx2", cam2_init.K[0], 50, 5000);
    pangolin::Var<double> fy2("ui.fy2", cam2_init.K[1], 50, 5000);
    pangolin::Var<double> cx2("ui.cx2", cam2_init.K[2], 0, 4000);
    pangolin::Var<double> cy2("ui.cy2", cam2_init.K[3], 0, 4000);
    pangolin::Var<double> k2_1("ui.k2_1", cam2_init.D[0], -2, 2);
    pangolin::Var<double> k2_2("ui.k2_2", cam2_init.D[1], -2, 2);
    pangolin::Var<double> k2_3("ui.k2_3", cam2_init.D[2], -2, 2);
    pangolin::Var<double> k2_4("ui.k2_4", cam2_init.D[3], -2, 2);

    // Inter-camera extrinsics (cam1->cam0, cam2->cam0)
    pangolin::Var<double> c10_r("ui.cam1_r", cam1_to_cam0_init.rpy.x(), -M_PI, M_PI);
    pangolin::Var<double> c10_p("ui.cam1_p", cam1_to_cam0_init.rpy.y(), -M_PI, M_PI);
    pangolin::Var<double> c10_y("ui.cam1_y", cam1_to_cam0_init.rpy.z(), -M_PI, M_PI);
    pangolin::Var<double> c10_tx("ui.cam1_tx", cam1_to_cam0_init.t.x(), -2, 2);
    pangolin::Var<double> c10_ty("ui.cam1_ty", cam1_to_cam0_init.t.y(), -2, 2);
    pangolin::Var<double> c10_tz("ui.cam1_tz", cam1_to_cam0_init.t.z(), -2, 2);

    pangolin::Var<double> c20_r("ui.cam2_r", cam2_to_cam0_init.rpy.x(), -M_PI, M_PI);
    pangolin::Var<double> c20_p("ui.cam2_p", cam2_to_cam0_init.rpy.y(), -M_PI, M_PI);
    pangolin::Var<double> c20_y("ui.cam2_y", cam2_to_cam0_init.rpy.z(), -M_PI, M_PI);
    pangolin::Var<double> c20_tx("ui.cam2_tx", cam2_to_cam0_init.t.x(), -2, 2);
    pangolin::Var<double> c20_ty("ui.cam2_ty", cam2_to_cam0_init.t.y(), -2, 2);
    pangolin::Var<double> c20_tz("ui.cam2_tz", cam2_to_cam0_init.t.z(), -2, 2);

    // Board pose (target->cam0) — this is per frame; we initialize to the given frame value on change
    pangolin::Var<double> b_r("ui.board_r", 0, -M_PI, M_PI);
    pangolin::Var<double> b_p("ui.board_p", 0, -M_PI, M_PI);
    pangolin::Var<double> b_y("ui.board_y", 0, -M_PI, M_PI);
    pangolin::Var<double> b_tx("ui.board_tx", 0, -5, 5);
    pangolin::Var<double> b_ty("ui.board_ty", 0, -5, 5);
    pangolin::Var<double> b_tz("ui.board_tz", 1.0, 0.01, 20.0);
    pangolin::Var<bool>  reset_pose_from_frame("ui.reset_pose_from_frame", true, false);

    // Error readouts
    pangolin::Var<double> err0("ui.error_cam0", 0.0);
    pangolin::Var<double> err1("ui.error_cam1", 0.0);
    pangolin::Var<double> err2("ui.error_cam2", 0.0);
    pangolin::Var<double> errTot("ui.error_total", 0.0);

    // Keyboard stepping
    pangolin::RegisterKeyPressCallback('a', [&]() {
        frame_idx = std::max(0, (int)frame_idx.Get() - 1);
        prev_f = true;
    });
    pangolin::RegisterKeyPressCallback('d', [&]() {
        frame_idx = std::min((int)N-1, (int)frame_idx.Get() + 1);
        next_f = true;
    });

    auto set_board_sliders_from_frame = [&](int idx) {
        if (idx < 0 || idx >= (int)board_poses_init.size()) return;
        b_r  = board_poses_init[idx].rpy.x();
        b_p  = board_poses_init[idx].rpy.y();
        b_y  = board_poses_init[idx].rpy.z();
        b_tx = board_poses_init[idx].t.x();
        b_ty = board_poses_init[idx].t.y();
        b_tz = board_poses_init[idx].t.z();
    };
    set_board_sliders_from_frame(frame_idx);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        const int i = std::clamp((int)frame_idx.Get(), 0, (int)N-1);
        if (reset_pose_from_frame.GuiChanged() || prev_f.GuiChanged() || next_f.GuiChanged()) {
            set_board_sliders_from_frame(i);
            reset_pose_from_frame = false;
            prev_f = false;
            next_f = false;
        }

        // Build intrinsics/dist
        double K0[4] = {fx0, fy0, cx0, cy0};
        double D0[4] = {k0_1, k0_2, k0_3, k0_4};

        double K1[4] = {fx1, fy1, cx1, cy1};
        double D1[4] = {k1_1, k1_2, k1_3, k1_4};

        double K2[4] = {fx2, fy2, cx2, cy2};
        double D2[4] = {k2_1, k2_2, k2_3, k2_4};

        // Poses
        Eigen::Matrix3d R0 = R_from_rpy(Eigen::Vector3d(b_r, b_p, b_y));
        Eigen::Vector3d t0(b_tx, b_ty, b_tz);

        Eigen::Matrix3d R10 = R_from_rpy(Eigen::Vector3d(c10_r, c10_p, c10_y));
        Eigen::Vector3d t10(c10_tx, c10_ty, c10_tz);

        Eigen::Matrix3d R20 = R_from_rpy(Eigen::Vector3d(c20_r, c20_p, c20_y));
        Eigen::Vector3d t20(c20_tx, c20_ty, c20_tz);

        // Fetch observations (may be empty)
        const std::vector<Eigen::Vector2d>* obs0 = (i < (int)frames_cam0.size() ? &frames_cam0[i].observations : nullptr);
        const std::vector<Eigen::Vector2d>* obs1 = (i < (int)frames_cam1.size() ? &frames_cam1[i].observations : nullptr);
        const std::vector<Eigen::Vector2d>* obs2 = (i < (int)frames_cam2.size() ? &frames_cam2[i].observations : nullptr);

        // Project for each camera
        std::vector<Eigen::Vector2d> proj0, proj1, proj2;
        proj0.reserve(board_points.size());
        proj1.reserve(board_points.size());
        proj2.reserve(board_points.size());

        for (size_t j = 0; j < board_points.size(); ++j) {
            // target -> cam0
            Eigen::Vector3d Xc0 = ToCam0(R0, t0, board_points[j]);

            // cam0 projection
            Eigen::Vector2d u0; bool ok0 = ProjectFisheyeKB(Xc0, K0, D0, u0);
            if (!ok0) u0 = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN());
            proj0.push_back(u0);

            // target -> cam1
            Eigen::Vector3d Xc1 = ToCamX_fromCam0(R10, t10, Xc0);
            Eigen::Vector2d u1; bool ok1 = ProjectFisheyeKB(Xc1, K1, D1, u1);
            if (!ok1) u1 = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN());
            proj1.push_back(u1);

            // target -> cam2
            Eigen::Vector3d Xc2 = ToCamX_fromCam0(R20, t20, Xc0);
            Eigen::Vector2d u2; bool ok2 = ProjectFisheyeKB(Xc2, K2, D2, u2);
            if (!ok2) u2 = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                           std::numeric_limits<double>::quiet_NaN());
            proj2.push_back(u2);
        }

        // Compute errors (RMS) for frames that have observations
        double e0 = 0, e1 = 0, e2 = 0, etot = 0;
        size_t cams_count = 0;

        if (obs0 && obs0->size() == proj0.size() && !obs0->empty()) {
            e0 = ComputeRMSError(*obs0, proj0);
            err0 = e0; etot += e0; cams_count++;
        } else { err0 = 0.0; }

        if (obs1 && obs1->size() == proj1.size() && !obs1->empty()) {
            e1 = ComputeRMSError(*obs1, proj1);
            err1 = e1; etot += e1; cams_count++;
        } else { err1 = 0.0; }

        if (obs2 && obs2->size() == proj2.size() && !obs2->empty()) {
            e2 = ComputeRMSError(*obs2, proj2);
            err2 = e2; etot += e2; cams_count++;
        } else { err2 = 0.0; }

        errTot = (cams_count ? etot / cams_count : 0.0);

        // ----------------- Draw 2D panels side-by-side -----------------
        auto draw_points = [](const std::vector<Eigen::Vector2d>& pts, float xoff) {
            glBegin(GL_POINTS);
            for (const auto& p : pts) {
                if (!std::isfinite(p.x()) || !std::isfinite(p.y())) continue;
                glVertex2f((float)(p.x() + xoff), (float)p.y());
            }
            glEnd();
        };
        auto draw_pairs = [](const std::vector<Eigen::Vector2d>& obs,
                             const std::vector<Eigen::Vector2d>& prj,
                             float xoff) {
            const size_t n = std::min(obs.size(), prj.size());
            glBegin(GL_LINES);
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(prj[i].x()) || !std::isfinite(prj[i].y())) continue;
                glVertex2f((float)(obs[i].x() + xoff), (float)obs[i].y());
                glVertex2f((float)(prj[i].x() + xoff), (float)prj[i].y());
            }
            glEnd();
        };

        const float x0 = 0.0f;
        const float x1 = (float)panel_w;
        const float x2 = (float)(2*panel_w);

        glPointSize(4.0f);

        // Cam0
        if (obs0) {
            glColor3f(0.0f, 1.0f, 0.0f); // observed
            draw_points(*obs0, x0);
        }
        glColor3f(1.0f, 0.0f, 0.0f); // projected
        draw_points(proj0, x0);

        glColor3f(1.0f, 1.0f, 0.0f); // error lines
        if (obs0) draw_pairs(*obs0, proj0, x0);

        // Cam1
        if (obs1) {
            glColor3f(0.0f, 1.0f, 0.0f);
            draw_points(*obs1, x1);
        }
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_points(proj1, x1);

        glColor3f(1.0f, 1.0f, 0.0f);
        if (obs1) draw_pairs(*obs1, proj1, x1);

        // Cam2
        if (obs2) {
            glColor3f(0.0f, 1.0f, 0.0f);
            draw_points(*obs2, x2);
        }
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_points(proj2, x2);

        glColor3f(1.0f, 1.0f, 0.0f);
        if (obs2) draw_pairs(*obs2, proj2, x2);

        pangolin::FinishFrame();
    }
}


using TimestampList = std::vector<uint64_t>;

int main(int argc, char** argv) {
    std::cout << "Argument count (argc): " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }
    if (argc < 2) {
        std::cerr << "Usage: ./calibrate_fisheye_camera <data_file.csv>" << std::endl;
        return -1;
    }

    std::string data_file = argv[1];

    // Load the AprilTag board geometry

    std::vector<Eigen::Vector3d> board_points;
    board_points = loadAprilTagBoardFlat("/home/jake/calibration_w_eigan/apiril_tag_board.json");

    // Step 1: Load and process CSV data for all cameras
    auto [obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0] = processCSV(data_file, 0);
    auto [obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1] = processCSV(data_file, 1);
    auto [obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2] = processCSV(data_file, 2);

    // Load K_0, K_1, K_2;
    Eigen::Matrix3d K_0;
    Eigen::Matrix3d K_1;
    Eigen::Matrix3d K_2;


    


    std::cin.get();  // Wait for user input before proceeding


    // Initialize camera parameters
    // Eigen::Matrix3d K_0;
    // K_0 << 800, 0, 640,
    //        0, 800, 480,
    //        0, 0, 1;
    // Eigen::Matrix3d K_1;
    // K_1 << 810, 0, 650,
    //        0, 810, 470,
    //        0, 0, 1;
    // Eigen::Matrix3d K_2;
    // K_2 << 820, 0, 660,
    //        0, 820, 460,
    //        0, 0, 1;




    double intrinsic_0[4], dist_0[4];
    double intrinsic_1[4], dist_1[4];
    double intrinsic_2[4], dist_2[4];
    std::vector<std::array<double, 7>> extrinsics_0, extrinsics_1, extrinsics_2;
    std::vector<double> timestamps_0, timestamps_1, timestamps_2;
    double rvec_cam_1[3], tvec_cam_1[3];
    double rvec_cam_2[3], tvec_cam_2[3];

    if (!LoadCalibrationResult("/home/jake/calibration_w_eigan/calibration_output.json",
        intrinsic_0, dist_0, extrinsics_0, timestamps_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        rvec_cam_1, tvec_cam_1, rvec_cam_2, tvec_cam_2))
    {
        return -1;
    }

    // Now you can build Eigen::Matrix3d K_0, K_1, K_2 from intrinsics:
    Eigen::Matrix3d K_0;
    K_0 << intrinsic_0[0], 0, intrinsic_0[2],
            0, intrinsic_0[1], intrinsic_0[3],
            0, 0, 1;
    Eigen::Matrix3d K_1;
    K_1 << intrinsic_1[0], 0, intrinsic_1[2],
            0, intrinsic_1[1], intrinsic_1[3],
            0, 0, 1;
    Eigen::Matrix3d K_2;
    K_2 << intrinsic_2[0], 0, intrinsic_2[2],
            0, intrinsic_2[1], intrinsic_2[3],
            0, 0, 1;

    // double intrinsic_0[4] = {K_0(0, 0), K_0(1, 1), K_0(0, 2), K_0(1, 2)};
    // double dist_0[4] = {-0.013, -0.02, 0.063, -0.03}; 
    // double intrinsic_1[4] = {K_1(0, 0), K_1(1, 1), K_1(0, 2), K_1(1, 2)};
    // double dist_1[4] = {0.09, -0.057, 0.014, -0.004}; 
    // double intrinsic_2[4] = {K_2(0, 0), K_2(1, 1), K_2(0, 2), K_2(1, 2)};
    // double dist_2[4] = {0.03, -0.065, 0.065, -0.0034}; 

    

    // --- START: single-target-pose initialization (drop-in) --------------------

    auto quatTransToMatrix = [](const double quat_arr[4], const double t_arr[3]) {
        Eigen::Quaterniond q(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3]); // [w,x,y,z]
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T.block<3,1>(0,3) = Eigen::Vector3d(t_arr[0], t_arr[1], t_arr[2]);
        return T;
    };

    auto arrayPoseToMatrix = [](const std::array<double,7>& a) {
        Eigen::Quaterniond q(a[0], a[1], a[2], a[3]); // [w,x,y,z]
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T.block<3,1>(0,3) = Eigen::Vector3d(a[4], a[5], a[6]);
        return T;
    };

    auto matrixToArrayPose = [](const Eigen::Matrix4d& T) {
        Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::Quaterniond q(R);
        std::array<double,7> a;
        a[0] = q.w(); a[1] = q.x(); a[2] = q.y(); a[3] = q.z();
        a[4] = T(0,3); a[5] = T(1,3); a[6] = T(2,3);
        return a;
    };

    // Build target_poses (one pose per master_timestamps entry) in Camera 0 frame
    std::vector<std::array<double,7>> target_poses;
    target_poses.reserve(master_timestamps.size());

    // Build initial camX_in_cam0 transforms from the user-provided qvec_cam_X/tvec_cam_X
    // (these variables are declared later in your original main; if you want to
    // keep the same order, you can move this block down after you construct qvec_cam_1/qvec_cam_2.
    // For clarity we assume qvec_cam_1/qvec_cam_2 and tvec_cam_1/tvec_cam_2 are available here.)
    //
    // If they are not yet available at this exact insertion point, move the following
    // two lines (construction of cam1_in_cam0/cam2_in_cam0) down to right after you
    // compute qvec_cam_1/tvec_cam_1 and qvec_cam_2/tvec_cam_2.

    double cam1_quat_tmp[4]; double cam2_quat_tmp[4];
    double cam1_trans_tmp[3]; double cam2_trans_tmp[3];
    // We'll copy user variables into temporaries if they exist; if they don't exist yet,
    // the below initialization will be overwritten later. To keep this block safe for direct
    // paste, we check and set identity fallback here:
    for (int i=0;i<4;++i) { cam1_quat_tmp[i] = (i==0?1.0:0.0); cam2_quat_tmp[i] = (i==0?1.0:0.0); }
    for (int i=0;i<3;++i) { cam1_trans_tmp[i] = 0.0; cam2_trans_tmp[i] = 0.0; }

    // NOTE: if qvec_cam_1/qvec_cam_2 and tvec_cam_1/tvec_cam_2 are declared *after* this point
    // in your main (as in your posted main), we'll re-create the camX_in_cam0 matrices again
    // after those quaternions are available — see the second construction below.

    // Build target_poses now using available extrinsics_* lists
    for (const auto& entry : master_timestamps) {
        if (entry.cam0_idx != -1) {
            // Use camera 0's observed target pose directly (pose is target_in_cam0)
            target_poses.push_back(extrinsics_0[entry.cam0_idx]);

        } else if (entry.cam1_idx != -1) {
            // Camera 1 saw it: convert target_in_cam1 -> target_in_cam0
            Eigen::Matrix4d T_target_in_cam1 = arrayPoseToMatrix(extrinsics_1[entry.cam1_idx]);

            // If we don't yet have the real cam1_in_cam0 (quaternion + t will be set later),
            // assume identity for now; the optimizer init will still be meaningful.
            Eigen::Matrix4d T_cam1_in_cam0 = Eigen::Matrix4d::Identity();
            // If user has provided qvec_cam_1 and tvec_cam_1 earlier, use them:
            // (we can't refer to variables that may not exist here without breaking drop-in,
            // so we'll recompute again later right after qvec_cam_1/tvec_cam_1 are available.)
            Eigen::Matrix4d T_target_in_cam0 = T_cam1_in_cam0 * T_target_in_cam1;
            target_poses.push_back(matrixToArrayPose(T_target_in_cam0));

        } else if (entry.cam2_idx != -1) {
            // Camera 2 saw it: convert target_in_cam2 -> target_in_cam0
            Eigen::Matrix4d T_target_in_cam2 = arrayPoseToMatrix(extrinsics_2[entry.cam2_idx]);

            Eigen::Matrix4d T_cam2_in_cam0 = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d T_target_in_cam0 = T_cam2_in_cam0 * T_target_in_cam2;
            target_poses.push_back(matrixToArrayPose(T_target_in_cam0));

        } else {
            // No camera saw the target at this timestamp — default to identity
            target_poses.push_back({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            std::cerr << "Warning: No camera saw target at timestamp " << entry.timestamp_id
                      << ", using identity pose." << std::endl;
        }
    }

    // --- If qvec_cam_1/qvec_cam_2 and tvec_cam_1/tvec_cam_2 are declared later in main
    //     (as in your existing main), reconstruct any target_poses that were bootstrapped
    //     from camera1/camera2 using the real initial cam transforms now that those
    //     variables exist. This keeps the paste-drop safe and produces correct bootstrapping.
    //
    // Rebuild cam transforms from the actual user variables (overwrite temporaries):
    {
        // Build real cam1_in_cam0 and cam2_in_cam0 using the variables you define later.
        // If those variables are located after this insertion point, move this small block
        // to just after you call ceres::AngleAxisToQuaternion(...) for cam1 and cam2.
        Eigen::Matrix4d cam1_in_cam0 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d cam2_in_cam0 = Eigen::Matrix4d::Identity();

        // Only build if qvec_cam_1 and tvec_cam_1 are in scope (they are later in your main).
        // To be safe, check symbol existence at compile time isn't possible here; simply
        // re-run this population after qvec_cam_1/tvec_cam_1 are assigned in your main
        // (move these three lines down if needed).
        // Example (if in scope):
        // cam1_in_cam0 = quatTransToMatrix(qvec_cam_1, tvec_cam_1);
        // cam2_in_cam0 = quatTransToMatrix(qvec_cam_2, tvec_cam_2);

        // Now, **recompute** any target_poses that were created from extrinsics_1/extrinsics_2
        for (size_t i = 0; i < master_timestamps.size(); ++i) {
            const auto &entry = master_timestamps[i];
            if (entry.cam0_idx != -1) continue; // already a cam0 measurement, keep it

            if (entry.cam1_idx != -1) {
                // recompute using cam1_in_cam0 (if cam1_in_cam0 is identity because you left it,
                // result is same as before)
                Eigen::Matrix4d T_target_in_cam1 = arrayPoseToMatrix(extrinsics_1[entry.cam1_idx]);
                Eigen::Matrix4d T_target_in_cam0 = cam1_in_cam0 * T_target_in_cam1;
                target_poses[i] = matrixToArrayPose(T_target_in_cam0);

            } else if (entry.cam2_idx != -1) {
                Eigen::Matrix4d T_target_in_cam2 = arrayPoseToMatrix(extrinsics_2[entry.cam2_idx]);
                Eigen::Matrix4d T_target_in_cam0 = cam2_in_cam0 * T_target_in_cam2;
                target_poses[i] = matrixToArrayPose(T_target_in_cam0);
            }
        }
    }

    // Debug print (optional)
    std::cout << "Initialized " << target_poses.size() << " target_poses (in cam0 frame)." << std::endl;

    VisualizeStereoReprojectionTuner(
    // AprilTag board geometry (object points in board/target frame; order must match detections)
    const std::vector<Eigen::Vector3d>& board_points,

    // Per-frame detections for each camera (same indexing across cams; empty vectors allowed when not detected)
    const std::vector<FrameDetections>& frames_cam0,
    const std::vector<FrameDetections>& frames_cam1,
    const std::vector<FrameDetections>& frames_cam2,

    // Initial intrinsics & fisheye params for each camera
    const CameraInit& cam0_init,
    const CameraInit& cam1_init,
    const CameraInit& cam2_init,

    // Initial inter-camera transforms (cam1->cam0, cam2->cam0)
    const InterCamInit& cam1_to_cam0_init,
    const InterCamInit& cam2_to_cam0_init,

    // Initial board poses target->cam0 per frame (same size as frames)
    const std::vector<BoardPoseInit>& board_poses_init
    ) {

    return 0;  // Return early to avoid running the rest of the main
}