#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iomanip>
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


std::unordered_map<int, Eigen::Vector3d> loadAprilTagBoardFlat(const std::string& json_file) {
    std::ifstream file(json_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open JSON file: " + json_file);
    }

    json board_config;
    file >> board_config;

    int tag_cols = board_config["tagCols"];
    int tag_rows = board_config["tagRows"];
    double tag_size = board_config["tagSize"];
    double tag_spacing = board_config["tagSpacing"];

    std::unordered_map<int, Eigen::Vector3d> id_to_point;

    int tag_id = 0;
    for (int row = 0; row < tag_rows; ++row) {
        for (int col = 0; col < tag_cols; ++col) {
            double tag_x = col * (tag_size + tag_spacing);
            double tag_y = row * (tag_size + tag_spacing);
            double tag_z = 0.0;

            Eigen::Vector3d corners[4] = {
                {tag_x, tag_y, tag_z},                                      // Top-left
                {tag_x + tag_size, tag_y, tag_z},                          // Top-right
                {tag_x + tag_size, tag_y + tag_size, tag_z},               // Bottom-right
                {tag_x, tag_y + tag_size, tag_z}                           // Bottom-left
            };

            for (int i = 0; i < 4; ++i) {
                int corner_id = tag_id * 4 + i;
                id_to_point[corner_id] = corners[i];
            }

            ++tag_id;
        }
    }

    return id_to_point;
}






std::tuple<HomographyList, TimestampList> computeHomographies(
    const std::vector<Point3dVec>& obj_pts_list,
    const std::vector<Point2dVec>& img_pts_list,
    const TimestampList& timestamp_list)
{
    HomographyList homographies;
    TimestampList filtered_timestamps;

    for (size_t k = 0; k < obj_pts_list.size(); ++k) {
        const auto& obj_pts = obj_pts_list[k];
        const auto& img_pts = img_pts_list[k];
        double timestamp = timestamp_list[k];

        if (obj_pts.size() != img_pts.size() || obj_pts.size() < 4) {
            std::cerr << "Skipping due to insufficient or mismatched points." << std::endl;
            continue;
        }

        Eigen::MatrixXd A(2 * obj_pts.size(), 9);
        for (size_t i = 0; i < obj_pts.size(); ++i) {
            double X = obj_pts[i].x(), Y = obj_pts[i].y();
            double x = img_pts[i].x(), y = img_pts[i].y();

            A.row(2 * i)     << -X, -Y, -1,  0,  0,  0,  x * X, x * Y, x;
            A.row(2 * i + 1) <<  0,  0,  0, -X, -Y, -1,  y * X, y * Y, y;
        }

        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd h = svd.matrixV().col(8);  // Last column of V

        // Check condition number
        double min_singular = svd.singularValues()(svd.singularValues().size() - 1);
        if (min_singular < 1e-8) {
            std::cerr << "Singular values too small, skipping homography." << std::endl;
            continue;
        }

        // Reshape into 3x3 matrix
        Eigen::Matrix3d H;
        H << h(0), h(1), h(2),
             h(3), h(4), h(5),
             h(6), h(7), h(8);

        H /= H(2, 2);  // Normalize

        homographies.push_back(H);
        filtered_timestamps.push_back(timestamp);
    }

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
    std::vector<Point3dVec> filtered_obj_pts;
    std::vector<Point2dVec> filtered_img_pts;
    std::vector<IDVec> filtered_corner_ids;

    // Use a set for faster lookup
    std::unordered_set<double> timestamp_set(filtered_timestamps.begin(), filtered_timestamps.end());

    for (size_t i = 0; i < timestamp_list.size(); ++i) {
        if (timestamp_set.count(timestamp_list[i]) > 0) {
            filtered_obj_pts.push_back(obj_pts_list[i]);
            filtered_img_pts.push_back(img_pts_list[i]);
            filtered_corner_ids.push_back(corner_ids_list[i]);
        }
    }

    return {filtered_obj_pts, filtered_img_pts, filtered_corner_ids};
}


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


Eigen::Matrix3d compute_intrinsic_params(const std::vector<Eigen::Matrix3d>& H_list)
{
    // std::cout << "H_list" << std::endl;
    // for (const auto& H : H_list) {
    //     std::cout << H << std::endl;
    // }
    std::vector<Eigen::Matrix<double, 6, 1>> V;

    for (const auto& H : H_list) {
        Eigen::Vector3d h1 = H.col(0);
        Eigen::Vector3d h2 = H.col(1);

        Eigen::Matrix<double, 6, 1> v12;
        v12 << h1(0) * h2(0),
               h1(0) * h2(1) + h1(1) * h2(0),
               h1(1) * h2(1),
               h1(2) * h2(0) + h1(0) * h2(2),
               h1(2) * h2(1) + h1(1) * h2(2),
               h1(2) * h2(2);

        Eigen::Matrix<double, 6, 1> v11_minus_v22;
        v11_minus_v22 << h1(0)*h1(0) - h2(0)*h2(0),
                         2*(h1(0)*h1(1) - h2(0)*h2(1)),
                         h1(1)*h1(1) - h2(1)*h2(1),
                         2*(h1(0)*h1(2) - h2(0)*h2(2)),
                         2*(h1(1)*h1(2) - h2(1)*h2(2)),
                         h1(2)*h1(2) - h2(2)*h2(2);

        V.push_back(v12);
        V.push_back(v11_minus_v22);
    }

    // Stack into a matrix
    Eigen::MatrixXd V_mat(V.size(), 6);
    for (size_t i = 0; i < V.size(); ++i) {
        V_mat.row(i) = V[i].transpose();
    }

    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V_mat, Eigen::ComputeFullV);
    // std::cout << "SVD singular values: " << svd.singularValues().transpose() << std::endl;
    Eigen::VectorXd b = svd.matrixV().col(5);  // Last column of V
    // std::cout << "b: " << b.transpose() << std::endl;

    // Form B matrix
    Eigen::Matrix3d B;
    B << b(0), b(1), b(3),
         b(1), b(2), b(4),
         b(3), b(4), b(5);
    
    // std::cout << "B: " << B << std::endl;
    double v0 = (B(0,1)*B(0,2) - B(1,2)*B(0,0)) / (B(0,0)*B(1,1) - B(0,1)*B(0,1));
    double lambda = B(2,2) - (B(0,2)*B(0,2) + v0*(B(0,1)*B(0,2) - B(1,2)*B(0,0))) / B(0,0);
    double alpha = std::sqrt(lambda / B(0,0));
    double beta  = std::sqrt(lambda * B(0,0) / (B(0,0)*B(1,1) - B(0,1)*B(0,1)));
    double gamma = -B(0,1) * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B(0,2) * alpha * alpha / lambda;
    // std::cout << "alpha: " << alpha << ", beta: " << beta
    //           << ", gamma: " << gamma << ", u0: " << u0
    //           << ", v0: " << v0 << std::endl;



    Eigen::Matrix3d K;
    K << alpha, gamma, u0,
         0,     beta,  v0,
         0,     0,     1;
    // std::cin.get();  // Pause for debugging

    return K;
}

struct OptimizationFlags {
    bool optimize_intrinsics = true;
    bool optimize_distortion = true;
    bool optimize_inter_camera = true;
    bool optimize_target_poses = true;
};

Eigen::Matrix3d robust_intrinsic_estimation(
    const std::vector<Eigen::Matrix3d>& H_list,
    int max_trials = 10,
    int min_h_required = 3)
{
    auto has_nan = [](const Eigen::Matrix3d& K) {
        return !K.allFinite();
    };

    // First try with the full list
    Eigen::Matrix3d K_full = compute_intrinsic_params(H_list);
    if (!has_nan(K_full)) {
        // std::cout << "Recovered K using full H_list." << std::endl;
        return K_full;
    }

    // Otherwise, try randomized subsets
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<Eigen::Matrix3d> valid_Ks;

    for (int trial = 0; trial < max_trials; ++trial) {
        int subset_size = std::min<int>(H_list.size(), min_h_required + trial % 3);
        std::vector<Eigen::Matrix3d> subset;
        std::sample(H_list.begin(), H_list.end(),
                    std::back_inserter(subset),
                    subset_size, gen);

        Eigen::Matrix3d K = compute_intrinsic_params(subset);
        if (!has_nan(K)) {
            std::cout << "Recovered K from trial " << trial << " with subset size " << subset.size() << "." << std::endl;
            valid_Ks.push_back(K);
        }
    }

    if (!valid_Ks.empty()) {
        // Average valid Ks
        Eigen::Matrix3d K_avg = Eigen::Matrix3d::Zero();
        for (const auto& K : valid_Ks) {
            K_avg += K;
        }
        K_avg /= static_cast<double>(valid_Ks.size());
        std::cout << "Returning average of " << valid_Ks.size() << " valid K matrices." << std::endl;
        return K_avg;
    }

    std::cerr << "Failed to compute any valid K matrix after " << max_trials << " trials." << std::endl;
    return Eigen::Matrix3d::Identity();  // fallback or throw
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> compute_extrinsic_params(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K)
{
    // std::cout << "H: " << H << std::endl;
    // std::cout << "K: " << K << std::endl;
    Eigen::Matrix3d K_inv = K.inverse();

    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    double lambda = 1.0 / (K_inv * h1).norm();

    Eigen::Vector3d r1 = lambda * (K_inv * h1);
    Eigen::Vector3d r2 = lambda * (K_inv * h2);
    Eigen::Vector3d t  = lambda * (K_inv * h3);
    Eigen::Vector3d r3 = r1.cross(r2);

    Eigen::Matrix3d R;
    R.col(0) = r1;
    R.col(1) = r2;
    R.col(2) = r3;

    // Re-orthonormalize R using SVD to ensure it's a valid rotation matrix
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    // std::cout << "R: " << R << std::endl;
    // std::cout << "t: " << t.transpose() << std::endl;
    // std::cin.get();  // Pause for debugging

    return {R, t};
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



// A hand-rolled cost functor that:
// - takes intrinsics/distortion
// - takes target pose expressed in cam0 (qw,qx,qy,qz, tx,ty,tz)
// - takes camX pose expressed in cam0 (q_cam, t_cam) (maps camX -> cam0: X_cam0 = R_cam * X_camX + t_cam)
// - computes target->camX by transforming the 3D point through target->cam0 then into camX frame
// - projects via a 4-term Kannala-Brandt style model (theta distortion with 4 coefficients in 'dist')
struct FisheyeReproj_TargetInCam0 {
    FisheyeReproj_TargetInCam0(const Eigen::Vector2d& measured_px,
                                const Eigen::Vector3d& obj_pt)
        : measured_px_(measured_px), obj_pt_(obj_pt) {}

    template <typename T>
    bool operator()(const T* intrinsic, // fx, fy, cx, cy (size 4)
                    const T* dist,      // k1,k2,k3,k4 (size 4)  - Kannala-style
                    const T* target_q,  // qw,qx,qy,qz (size 4)
                    const T* target_t,  // tx,ty,tz (size 3)
                    const T* cam_q,     // qw,qx,qy,qz of camX_in_cam0 (size 4)
                    const T* cam_t,     // tx,ty,tz of camX_in_cam0 (size 3)
                    T* residuals) const
    {
        // //outout all arguments for debugging
        // std::cout << "intrinsic: " << intrinsic[0] << ", " << intrinsic[1] << ", "
        //           << intrinsic[2] << ", " << intrinsic[3] << std::endl;
        // std::cout << "dist: " << dist[0] << ", " << dist[1] << ", "
        //           << dist[2] << ", " << dist[3] << std::endl;
        // std::cout << "target_q: " << target_q[0] << ", " << target_q[1] << ", "
        //           << target_q[2] << ", " << target_q[3] << std::endl;
        // std::cout << "target_t: " << target_t[0] << ", " << target_t[1] << ", "
        //           << target_t[2] << std::endl;
        // std::cout << "cam_q: " << cam_q[0] << ", " << cam_q[1] << ", "
        //           << cam_q[2] << ", " << cam_q[3] << std::endl;
        // std::cout << "cam_t: " << cam_t[0] << ", " << cam_t[1] << ", "
        //           <<    cam_t[2] << std::endl;  
        // //output variables for debugging
        // std::cout << "obj_pt_: " << obj_pt_(0) << ", " << obj_pt_(1) << ", "
        //           << obj_pt_(2) << std::endl;
        // std::cout << "measured_px_: " << measured_px_(0) << ", " << measured_px_(1) << std::endl;
        // 1) compute 3D point in cam0: X_cam0 = R(target_q) * obj_pt + target_t
        T P_obj[3];
        P_obj[0] = T(obj_pt_(0));
        P_obj[1] = T(obj_pt_(1));
        P_obj[2] = T(obj_pt_(2));

        T P_cam0[3];
        ceres::QuaternionRotatePoint(target_q, P_obj, P_cam0);
        P_cam0[0] += target_t[0];
        P_cam0[1] += target_t[1];
        P_cam0[2] += target_t[2];

        // 2) transform into camX frame using inverse of camX_in_cam0:
        // camX_in_cam0 maps X_camX -> X_cam0: X_cam0 = R_cam * X_camX + t_cam
        // => X_camX = R_cam^T * (X_cam0 - t_cam)
        T cam_q_inv[4];
        cam_q_inv[0] = cam_q[0];
        cam_q_inv[1] = -cam_q[1];
        cam_q_inv[2] = -cam_q[2];
        cam_q_inv[3] = -cam_q[3];

        T diff[3];
        diff[0] = P_cam0[0] - cam_t[0];
        diff[1] = P_cam0[1] - cam_t[1];
        diff[2] = P_cam0[2] - cam_t[2];

        T P_camX[3];
        ceres::QuaternionRotatePoint(cam_q_inv, diff, P_camX); // rotate by R_cam^T

        // If point is behind camera, still compute residual (can optionally robustify)
        // 3) project using Kannala-Brandt style fisheye (4 coeffs). We follow:
        //    x = X/Z, y = Y/Z, r = sqrt(x^2+y^2), theta = atan(r)
        //    theta_d = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        //    scale = (r > eps) ? theta_d / r : 1
        //    x_p = scale*x; y_p = scale*y
        //    u = fx*x_p + cx, v = fy*y_p + cy

        T X = P_camX[0], Y = P_camX[1], Z = P_camX[2];
        const T eps = T(1e-12);
        T x = X / Z;
        T y = Y / Z;
        T r = ceres::sqrt(x*x + y*y);
        T theta = ceres::atan(r);

        T theta2 = theta*theta;
        T theta4 = theta2*theta2;
        T theta6 = theta4*theta2;
        T theta8 = theta4*theta4;

        T theta_d = theta * (T(1) + dist[0]*theta2 + dist[1]*theta4 + dist[2]*theta6 + dist[3]*theta8);

        T scale = r > eps ? theta_d / r : T(1.0); // if r~0, direction preserved

        T x_p = scale * x;
        T y_p = scale * y;

        T u = intrinsic[0] * x_p + intrinsic[2];
        T v = intrinsic[1] * y_p + intrinsic[3];

        // Clamp projection to image bounds if point is behind camera or outside image
        // This prevents optimizer from pushing points outside valid viewing area
        const T img_min_u = T(0.0);
        const T img_min_v = T(0.0);
        // Estimate image size from intrinsics (typically 2*cx width, 2*cy height)
        const T img_max_u = T(2.0) * intrinsic[2];  // approximate width
        const T img_max_v = T(2.0) * intrinsic[3];  // approximate height
        
        // Clamp projection to image bounds
        // For points behind camera (Z <= 0) or outside image, clamp to nearest border
        // This penalizes the optimizer for pushing points outside valid viewing area
        const T eps_behind = T(1e-9);  // threshold for "behind camera"
        const T obs_u = T(measured_px_(0));
        const T obs_v = T(measured_px_(1));
        
        // Check if behind camera: if Z <= eps_behind, use border farthest from observation (max penalty)
        // Otherwise, clamp to nearest border if outside image
        T behind_camera_penalty_u = (obs_u < (img_max_u + img_min_u) / T(2.0)) ? img_max_u : img_min_u;
        T behind_camera_penalty_v = (obs_v < (img_max_v + img_min_v) / T(2.0)) ? img_max_v : img_min_v;
        
        // Clamp u: if behind camera, use penalty border; otherwise clamp to nearest border
        T u_clamped = (Z <= eps_behind) ? behind_camera_penalty_u : 
                      ((u < img_min_u) ? img_min_u : ((u > img_max_u) ? img_max_u : u));
        
        // Clamp v: if behind camera, use penalty border; otherwise clamp to nearest border
        T v_clamped = (Z <= eps_behind) ? behind_camera_penalty_v :
                      ((v < img_min_v) ? img_min_v : ((v > img_max_v) ? img_max_v : v));
        
        // Compute residual using clamped projection
        // This penalizes points that would project outside the image
        residuals[0] = u_clamped - T(measured_px_(0));
        residuals[1] = v_clamped - T(measured_px_(1));
        
        // std::cout << "residuals (internal): " << residuals[0] << ", " << residuals[1] << std::endl;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& measured_px,
                                        const Eigen::Vector3d& obj_pt) {
        // parameters: intrinsic(4), dist(4), target_q(4), target_t(3), cam_q(4), cam_t(3)
        return (new ceres::AutoDiffCostFunction<FisheyeReproj_TargetInCam0, 2,
                                                4, 4, 4, 3, 4, 3>(
            new FisheyeReproj_TargetInCam0(measured_px, obj_pt)));
    }

    Eigen::Vector2d measured_px_;
    Eigen::Vector3d obj_pt_;
};


// Temporal smoothness residual compares two target poses (q,t) between consecutive frames.
// We'll use a simple AutoDiff functor that penalizes translation difference and quaternion
// difference (via angle between quaternions).
struct TempSmooth {
    TempSmooth(double trans_w, double rot_w) : tw(trans_w), rw(rot_w) {}

    template <typename T>
    bool operator()(const T* q1, const T* t1, const T* q2, const T* t2, T* residuals) const {
        // translation residuals
        residuals[0] = T(tw) * (t2[0] - t1[0]);
        residuals[1] = T(tw) * (t2[1] - t1[1]);
        residuals[2] = T(tw) * (t2[2] - t1[2]);

        // rotation residual: we use quaternion difference: qd = q2 * q1^{-1}
        T q1_inv[4] = { q1[0], -q1[1], -q1[2], -q1[3] };
        T qd[4];
        // quaternion multiply q2 * q1_inv
        qd[0] = q2[0]*q1_inv[0] - q2[1]*q1_inv[1] - q2[2]*q1_inv[2] - q2[3]*q1_inv[3];
        qd[1] = q2[0]*q1_inv[1] + q2[1]*q1_inv[0] + q2[2]*q1_inv[3] - q2[3]*q1_inv[2];
        qd[2] = q2[0]*q1_inv[2] - q2[1]*q1_inv[3] + q2[2]*q1_inv[0] + q2[3]*q1_inv[1];
        qd[3] = q2[0]*q1_inv[3] + q2[1]*q1_inv[2] - q2[2]*q1_inv[1] + q2[3]*q1_inv[0];

        // convert small quaternion difference to angle-axis approx: vector part ~ 0.5*angle*axis if qd ~= [1, vx, vy, vz]
        // We'll penalize the vector part scaled by rw
        residuals[3] = T(rw) * qd[1];
        residuals[4] = T(rw) * qd[2];
        residuals[5] = T(rw) * qd[3];
        return true;
    }

    double tw;
    double rw;
};

struct TimestampEntry {
    size_t timestamp_id;
    int cam0_idx;  // -1 if missing
    int cam1_idx;  // -1 if missing
    int cam2_idx;  // -1 if missing
};

// Load intrinsics and distortion from JSON file (same format as SaveCalibrationResult)
bool LoadIntrinsicsFromJson(const std::string& filepath,
                            double intrinsic_0[4], double dist_0[4],
                            double intrinsic_1[4], double dist_1[4],
                            double intrinsic_2[4], double dist_2[4]) {
    std::ifstream ifs(filepath);
    if (!ifs.good()) {
        std::cerr << "Failed to open intrinsics file: " << filepath << std::endl;
        return false;
    }
    
    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON file: " << e.what() << std::endl;
        return false;
    }
    
    // Helper to load camera intrinsics
    auto load_camera = [](const json& cam_json, double intrinsic[4], double dist[4]) -> bool {
        if (!cam_json.contains("intrinsics") || !cam_json.contains("distortion")) {
            return false;
        }
        
        auto intrin = cam_json["intrinsics"];
        auto dist_coeffs = cam_json["distortion"];
        
        if (intrin.size() != 4 || dist_coeffs.size() != 4) {
            return false;
        }
        
        for (int i = 0; i < 4; ++i) {
            intrinsic[i] = intrin[i].get<double>();
            dist[i] = dist_coeffs[i].get<double>();
        }
        
        return true;
    };
    
    bool success = true;
    if (j.contains("camera0")) {
        success = success && load_camera(j["camera0"], intrinsic_0, dist_0);
    } else {
        std::cerr << "JSON missing camera0 section" << std::endl;
        success = false;
    }
    
    if (j.contains("camera1")) {
        success = success && load_camera(j["camera1"], intrinsic_1, dist_1);
    } else {
        std::cerr << "JSON missing camera1 section" << std::endl;
        success = false;
    }
    
    if (j.contains("camera2")) {
        success = success && load_camera(j["camera2"], intrinsic_2, dist_2);
    } else {
        std::cerr << "JSON missing camera2 section" << std::endl;
        success = false;
    }
    
    if (success) {
        std::cout << "Successfully loaded intrinsics from: " << filepath << std::endl;
    }
    
    return success;
}

// Load optimization flags from JSON file
bool LoadOptimizationFlagsFromJson(const std::string& filepath,
                                    OptimizationFlags& flags) {
    std::ifstream ifs(filepath);
    if (!ifs.good()) {
        std::cerr << "Failed to open optimization flags file: " << filepath << std::endl;
        return false;
    }
    
    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON file: " << e.what() << std::endl;
        return false;
    }
    
    // Load flags (default to true if not present)
    if (j.contains("optimize_intrinsics")) {
        flags.optimize_intrinsics = j["optimize_intrinsics"].get<bool>();
    }
    if (j.contains("optimize_distortion")) {
        flags.optimize_distortion = j["optimize_distortion"].get<bool>();
    }
    if (j.contains("optimize_inter_camera")) {
        flags.optimize_inter_camera = j["optimize_inter_camera"].get<bool>();
    }
    if (j.contains("optimize_target_poses")) {
        flags.optimize_target_poses = j["optimize_target_poses"].get<bool>();
    }
    
    // std::cout << "Successfully loaded optimization flags from: " << filepath << std::endl;
    // std::cout << "  optimize_intrinsics: " << flags.optimize_intrinsics << std::endl;
    // std::cout << "  optimize_distortion: " << flags.optimize_distortion << std::endl;
    // std::cout << "  optimize_inter_camera: " << flags.optimize_inter_camera << std::endl;
    // std::cout << "  optimize_target_poses: " << flags.optimize_target_poses << std::endl;
    
    return true;
}

// Load inter-camera extrinsics from JSON file (same format as SaveCalibrationResult)
bool LoadExtrinsicsFromJson(const std::string& filepath,
                            double qvec_cam_1[4], double tvec_cam_1[3],
                            double qvec_cam_2[4], double tvec_cam_2[3]) {
    std::ifstream ifs(filepath);
    if (!ifs.good()) {
        std::cerr << "Failed to open extrinsics file: " << filepath << std::endl;
        return false;
    }
    
    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON file: " << e.what() << std::endl;
        return false;
    }
    
    if (!j.contains("inter_camera")) {
        std::cerr << "JSON missing inter_camera section" << std::endl;
        return false;
    }
    
    auto inter_cam = j["inter_camera"];
    
    // Load camera1_to_camera0
    if (!inter_cam.contains("camera1_to_camera0")) {
        std::cerr << "JSON missing camera1_to_camera0 section" << std::endl;
        return false;
    }
    
    auto cam1_to_cam0 = inter_cam["camera1_to_camera0"];
    if (!cam1_to_cam0.contains("quaternion") || !cam1_to_cam0.contains("translation_vector")) {
        std::cerr << "camera1_to_camera0 missing quaternion or translation_vector" << std::endl;
        return false;
    }
    
    auto q1 = cam1_to_cam0["quaternion"];
    auto t1 = cam1_to_cam0["translation_vector"];
    if (q1.size() != 4 || t1.size() != 3) {
        std::cerr << "Invalid quaternion/translation size for camera1_to_camera0" << std::endl;
        return false;
    }
    
    for (int i = 0; i < 4; ++i) {
        qvec_cam_1[i] = q1[i].get<double>();
    }
    for (int i = 0; i < 3; ++i) {
        tvec_cam_1[i] = t1[i].get<double>();
    }
    
    // Load camera2_to_camera0
    if (!inter_cam.contains("camera2_to_camera0")) {
        std::cerr << "JSON missing camera2_to_camera0 section" << std::endl;
        return false;
    }
    
    auto cam2_to_cam0 = inter_cam["camera2_to_camera0"];
    if (!cam2_to_cam0.contains("quaternion") || !cam2_to_cam0.contains("translation_vector")) {
        std::cerr << "camera2_to_camera0 missing quaternion or translation_vector" << std::endl;
        return false;
    }
    
    auto q2 = cam2_to_cam0["quaternion"];
    auto t2 = cam2_to_cam0["translation_vector"];
    if (q2.size() != 4 || t2.size() != 3) {
        std::cerr << "Invalid quaternion/translation size for camera2_to_camera0" << std::endl;
        return false;
    }
    
    for (int i = 0; i < 4; ++i) {
        qvec_cam_2[i] = q2[i].get<double>();
    }
    for (int i = 0; i < 3; ++i) {
        tvec_cam_2[i] = t2[i].get<double>();
    }
    
    // Normalize quaternions
    Eigen::Map<Eigen::Quaterniond> q1_map(qvec_cam_1);
    q1_map.normalize();
    Eigen::Map<Eigen::Quaterniond> q2_map(qvec_cam_2);
    q2_map.normalize();
    
    std::cout << "Successfully loaded extrinsics from: " << filepath << std::endl;
    return true;
}

void SaveCalibrationResult(
    const std::string& filename,
    const double intrinsic_0[4], const double dist_0[4],
    const double intrinsic_1[4], const double dist_1[4],
    const double intrinsic_2[4], const double dist_2[4],

    const double qvec_cam_1[4], const double tvec_cam_1[3],
    const double qvec_cam_2[4], const double tvec_cam_2[3],

    const std::vector<std::array<double, 7>>& target_poses, // target→world
    const std::vector<TimestampEntry>& master_timestamps,

    const std::vector<double>& frame_errors = {} // optional, length = N*3
) {
    json output;

    // --- Intrinsics & Distortion ---
    output["camera0"]["intrinsics"] = {intrinsic_0[0], intrinsic_0[1], intrinsic_0[2], intrinsic_0[3]};
    output["camera0"]["distortion"] = {dist_0[0], dist_0[1], dist_0[2], dist_0[3]};

    output["camera1"]["intrinsics"] = {intrinsic_1[0], intrinsic_1[1], intrinsic_1[2], intrinsic_1[3]};
    output["camera1"]["distortion"] = {dist_1[0], dist_1[1], dist_1[2], dist_1[3]};

    output["camera2"]["intrinsics"] = {intrinsic_2[0], intrinsic_2[1], intrinsic_2[2], intrinsic_2[3]};
    output["camera2"]["distortion"] = {dist_2[0], dist_2[1], dist_2[2], dist_2[3]};

    // --- Target poses (in world frame) ---
    for (size_t i = 0; i < target_poses.size(); ++i) {
        const auto& tp = target_poses[i];
        double timestamp = master_timestamps[i].timestamp_id;

        json pose;
        pose["timestamp"]   = timestamp;
        pose["quaternion"]  = {tp[0], tp[1], tp[2], tp[3]};
        pose["translation"] = {tp[4], tp[5], tp[6]};
        output["target_poses"].push_back(pose);
    }

    // --- Inter-Camera Transforms ---
    output["inter_camera"]["camera1_to_camera0"]["quaternion"]         = {qvec_cam_1[0], qvec_cam_1[1], qvec_cam_1[2], qvec_cam_1[3]};
    output["inter_camera"]["camera1_to_camera0"]["translation_vector"] = {tvec_cam_1[0], tvec_cam_1[1], tvec_cam_1[2]};

    output["inter_camera"]["camera2_to_camera0"]["quaternion"]         = {qvec_cam_2[0], qvec_cam_2[1], qvec_cam_2[2], qvec_cam_2[3]};
    output["inter_camera"]["camera2_to_camera0"]["translation_vector"] = {tvec_cam_2[0], tvec_cam_2[1], tvec_cam_2[2]};

    // --- Optional per-frame per-camera errors ---
    if (!frame_errors.empty()) {
        size_t N = master_timestamps.size();
        if (frame_errors.size() != N * 3) {
            std::cerr << "[SaveCalibrationResult] ERROR: frame_errors must be empty or size == timestamps*3 ("
                      << frame_errors.size() << " vs " << (N * 3) << ")\n";
        } else {
            for (size_t i = 0; i < N; ++i) {
                json err;
                err["cam0"] = frame_errors[i * 3 + 0];
                err["cam1"] = frame_errors[i * 3 + 1];
                err["cam2"] = frame_errors[i * 3 + 2];
                output["frame_errors"].push_back(err);
            }
        }
    }

    // --- Write to file ---
    std::ofstream ofs(filename);
    ofs << std::setw(4) << output << std::endl;
    // std::cout << "Saved calibration results to " << filename << std::endl;
}

// struct SaveIterationCallback : public ceres::IterationCallback {
//     const double* intrinsic_0;
//     const double* dist_0;
//     const double* intrinsic_1;
//     const double* dist_1;
//     const double* intrinsic_2;
//     const double* dist_2;
//     const double* qvec_cam_1;
//     const double* tvec_cam_1;
//     const double* qvec_cam_2;
//     const double* tvec_cam_2;
//     const std::vector<std::array<double, 7>>* target_poses;
//     const std::vector<TimestampEntry>* master_timestamps;
//     const std::string output_dir;
//     ceres::Problem* problem; // add this!

//     SaveIterationCallback(
//         const double* intrinsic_0, const double* dist_0,
//         const double* intrinsic_1, const double* dist_1,
//         const double* intrinsic_2, const double* dist_2,
//         const double* qvec_cam_1, const double* tvec_cam_1,
//         const double* qvec_cam_2, const double* tvec_cam_2,
//         const std::vector<std::array<double, 7>>* target_poses,
//         const std::vector<TimestampEntry>* master_timestamps,
//         const std::string& output_dir,
//         ceres::Problem* problem)
//         : intrinsic_0(intrinsic_0), dist_0(dist_0),
//           intrinsic_1(intrinsic_1), dist_1(dist_1),
//           intrinsic_2(intrinsic_2), dist_2(dist_2),
//           qvec_cam_1(qvec_cam_1), tvec_cam_1(tvec_cam_1),
//           qvec_cam_2(qvec_cam_2), tvec_cam_2(tvec_cam_2),
//           target_poses(target_poses), master_timestamps(master_timestamps),
//           output_dir(output_dir), problem(problem) {}

//     ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
//         std::string filename = output_dir + "/calibration_iter_" + std::to_string(summary.iteration) + ".json";

//         // Save camera parameters
//         SaveCalibrationResult(
//             filename,
//             intrinsic_0, dist_0,
//             intrinsic_1, dist_1,
//             intrinsic_2, dist_2,
//             qvec_cam_1, tvec_cam_1,
//             qvec_cam_2, tvec_cam_2,
//             *target_poses, *master_timestamps
//         );

//         // --- Evaluate residuals ---
//         ceres::Problem::EvaluateOptions eval_opts;
//         eval_opts.apply_loss_function = true;
//         std::vector<double> residuals;
//         double cost = 0.0;
//         problem->Evaluate(eval_opts, &cost, &residuals, nullptr, nullptr);

//         // Save residuals to a file
//         std::string res_file = output_dir + "/residuals_iter_" + std::to_string(summary.iteration) + ".txt";
//         std::ofstream fout(res_file);
//         for (double r : residuals)
//             fout << r << "\n";
//         fout.close();

//         return ceres::SOLVER_CONTINUE;
//     }
// };

class ResidualEvalCallback : public ceres::IterationCallback {
public:
    ResidualEvalCallback(
        const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0,
        const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0,
        const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1,
        const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1,
        const std::vector<std::vector<Eigen::Vector2d>>& img_pts_2,
        const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_2,
        const std::vector<TimestampEntry>& master_timestamps,
        std::vector<std::array<double,7>>* target_poses,
        double* intrinsic_0, double* dist_0,
        double* intrinsic_1, double* dist_1,
        double* intrinsic_2, double* dist_2,
        double* qvec_cam_1, double* tvec_cam_1,
        double* qvec_cam_2, double* tvec_cam_2,
        std::string log_dir)
        : img_pts_0_(img_pts_0), obj_pts_0_(obj_pts_0),
          img_pts_1_(img_pts_1), obj_pts_1_(obj_pts_1),
          img_pts_2_(img_pts_2), obj_pts_2_(obj_pts_2),
          master_timestamps_(master_timestamps),
          target_poses_(*target_poses),
          intrinsic_0_(intrinsic_0), dist_0_(dist_0),
          intrinsic_1_(intrinsic_1), dist_1_(dist_1),
          intrinsic_2_(intrinsic_2), dist_2_(dist_2),
          qvec_cam_1_(qvec_cam_1), tvec_cam_1_(tvec_cam_1),
          qvec_cam_2_(qvec_cam_2), tvec_cam_2_(tvec_cam_2),
          log_dir_(std::move(log_dir))
    {
        std::filesystem::create_directories(log_dir_);
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        double total_sq_err = 0.0;
        int total_pts = 0;

        // std::ofstream out(log_dir_ + "/iter_" + std::to_string(summary.iteration) + ".csv");
        // out << "cam,frame,point,res_u,res_v,res_norm\n";

        static const double cam0_q[4] = {1,0,0,0};
        static const double cam0_t[3] = {0,0,0};

        // Per-frame, per-camera error accumulator (N frames × 3 cameras)
        const size_t N = master_timestamps_.size();
        std::vector<double> frame_errors(N * 3, 0.0);

        // DEBUG: Output for frame 0 (or set DEBUG_FRAME to desired frame index)
        const size_t DEBUG_FRAME = 0;
        bool debug_frame = false;

        for (size_t idx = 0; idx < master_timestamps_.size(); ++idx) {
            const auto& entry = master_timestamps_[idx];
            const double* tq = target_poses_[idx].data();
            const double* tt = target_poses_[idx].data() + 4;

            // debug_frame = (idx == DEBUG_FRAME);
            debug_frame = false;
            
            if (debug_frame) {
                std::cout << "\n========== CALIB DEBUG: Frame " << idx << " (iteration " << summary.iteration << ") ==========" << std::endl;
                std::cout << "Timestamp ID: " << entry.timestamp_id << std::endl;
                std::cout << "Camera indices: cam0=" << entry.cam0_idx << ", cam1=" << entry.cam1_idx << ", cam2=" << entry.cam2_idx << std::endl;
                std::cout << "Target pose (target->cam0): q=[" << tq[0] << ", " << tq[1] << ", " << tq[2] << ", " << tq[3] 
                          << "], t=[" << tt[0] << ", " << tt[1] << ", " << tt[2] << "]" << std::endl;
                std::cout << "Cam0 intrinsics: K=[" << intrinsic_0_[0] << ", " << intrinsic_0_[1] << ", " << intrinsic_0_[2] << ", " << intrinsic_0_[3] << "]" << std::endl;
                std::cout << "Cam0 distortion: D=[" << dist_0_[0] << ", " << dist_0_[1] << ", " << dist_0_[2] << ", " << dist_0_[3] << "]" << std::endl;
                std::cout << "Cam1 intrinsics: K=[" << intrinsic_1_[0] << ", " << intrinsic_1_[1] << ", " << intrinsic_1_[2] << ", " << intrinsic_1_[3] << "]" << std::endl;
                std::cout << "Cam1 distortion: D=[" << dist_1_[0] << ", " << dist_1_[1] << ", " << dist_1_[2] << ", " << dist_1_[3] << "]" << std::endl;
                std::cout << "Cam1->Cam0 transform: q=[" << qvec_cam_1_[0] << ", " << qvec_cam_1_[1] << ", " << qvec_cam_1_[2] << ", " << qvec_cam_1_[3] 
                          << "], t=[" << tvec_cam_1_[0] << ", " << tvec_cam_1_[1] << ", " << tvec_cam_1_[2] << "]" << std::endl;
                std::cout << "Cam2 intrinsics: K=[" << intrinsic_2_[0] << ", " << intrinsic_2_[1] << ", " << intrinsic_2_[2] << ", " << intrinsic_2_[3] << "]" << std::endl;
                std::cout << "Cam2 distortion: D=[" << dist_2_[0] << ", " << dist_2_[1] << ", " << dist_2_[2] << ", " << dist_2_[3] << "]" << std::endl;
                std::cout << "Cam2->Cam0 transform: q=[" << qvec_cam_2_[0] << ", " << qvec_cam_2_[1] << ", " << qvec_cam_2_[2] << ", " << qvec_cam_2_[3] 
                          << "], t=[" << tvec_cam_2_[0] << ", " << tvec_cam_2_[1] << ", " << tvec_cam_2_[2] << "]" << std::endl;
            }

            double cam_err_sq[3] = {0.0, 0.0, 0.0};
            int cam_counts[3] = {0, 0, 0};

            // CAM0
            if (entry.cam0_idx != -1) {
                int i = entry.cam0_idx;
                if (debug_frame) {
                    // std::cout << "\n--- CAM0 Error Calculation (frame index " << i << ") ---" << std::endl;
                    // std::cout << "Number of points: " << img_pts_0_[i].size() << std::endl;
                }
                for (size_t j = 0; j < img_pts_0_[i].size(); ++j) {
                    FisheyeReproj_TargetInCam0 cost(img_pts_0_[i][j], obj_pts_0_[i][j]);
                    double res[2];
                    cost(intrinsic_0_, dist_0_, tq, tt, cam0_q, cam0_t, res);
                    double n2 = res[0]*res[0] + res[1]*res[1];
                    cam_err_sq[0] += n2;
                    cam_counts[0]++;
                    total_sq_err += n2;
                    total_pts++;

                    if (debug_frame && j < 10) {  // Print first 10 points
                        // std::cout << "  Point " << j << ": obs=[" << img_pts_0_[i][j].x() << ", " << img_pts_0_[i][j].y() 
                        //          << "], obj=[" << obj_pts_0_[i][j].x() << ", " << obj_pts_0_[i][j].y() << ", " << obj_pts_0_[i][j].z() 
                        //          << "], res=[" << res[0] << ", " << res[1] << "], res_norm=" << std::sqrt(n2) << std::endl;
                    }

                    // out << "cam0," << i << "," << j << ","
                    //     << res[0] << "," << res[1] << "," << std::sqrt(n2) << "\n";
                }
                if (debug_frame) {
                    double rms = (cam_counts[0] > 0) ? std::sqrt(cam_err_sq[0] / cam_counts[0]) : 0.0;
                    std::cout << "CAM0: " << cam_counts[0] << " points, Sum squared error: " << cam_err_sq[0] << ", RMS: " << rms << std::endl;
                }
            } else if (debug_frame) {
                std::cout << "\n--- CAM0: No observations ---" << std::endl;
            }

            // CAM1
            if (entry.cam1_idx != -1) {
                int i = entry.cam1_idx;
                if (debug_frame) {
                    // std::cout << "\n--- CAM1 Error Calculation (frame index " << i << ") ---" << std::endl;
                    // std::cout << "Number of points: " << img_pts_1_[i].size() << std::endl;
                }
                for (size_t j = 0; j < img_pts_1_[i].size(); ++j) {
                    FisheyeReproj_TargetInCam0 cost(img_pts_1_[i][j], obj_pts_1_[i][j]);
                    double res[2];
                    cost(intrinsic_1_, dist_1_, tq, tt, qvec_cam_1_, tvec_cam_1_, res);
                    double n2 = res[0]*res[0] + res[1]*res[1];
                    cam_err_sq[1] += n2;
                    cam_counts[1]++;
                    total_sq_err += n2;
                    total_pts++;

                    if (debug_frame && j < 10) {  // Print first 10 points
                        // std::cout << "  Point " << j << ": obs=[" << img_pts_1_[i][j].x() << ", " << img_pts_1_[i][j].y() 
                        //          << "], obj=[" << obj_pts_1_[i][j].x() << ", " << obj_pts_1_[i][j].y() << ", " << obj_pts_1_[i][j].z()
                        //          << "], res=[" << res[0] << ", " << res[1] << "], res_norm=" << std::sqrt(n2) << std::endl;
                    }

                    // out << "cam1," << i << "," << j << ","
                        // << res[0] << "," << res[1] << "," << std::sqrt(n2) << "\n";
                }
                if (debug_frame) {
                    double rms = (cam_counts[1] > 0) ? std::sqrt(cam_err_sq[1] / cam_counts[1]) : 0.0;
                    std::cout << "CAM1: " << cam_counts[1] << " points, Sum squared error: " << cam_err_sq[1] << ", RMS: " << rms << std::endl;
                }
            } else if (debug_frame) {
                std::cout << "\n--- CAM1: No observations ---" << std::endl;
            }

            // CAM2
            if (entry.cam2_idx != -1) {
                int i = entry.cam2_idx;
                if (debug_frame) {
                    // std::cout << "\n--- CAM2 Error Calculation (frame index " << i << ") ---" << std::endl;
                    // std::cout << "Number of points: " << img_pts_2_[i].size() << std::endl;
                }
                for (size_t j = 0; j < img_pts_2_[i].size(); ++j) {
                    FisheyeReproj_TargetInCam0 cost(img_pts_2_[i][j], obj_pts_2_[i][j]);
                    double res[2];
                    cost(intrinsic_2_, dist_2_, tq, tt, qvec_cam_2_, tvec_cam_2_, res);
                    double n2 = res[0]*res[0] + res[1]*res[1];
                    cam_err_sq[2] += n2;
                    cam_counts[2]++;
                    total_sq_err += n2;
                    total_pts++;

                    if (debug_frame && j < 10) {  // Print first 10 points
                        // std::cout << "  Point " << j << ": obs=[" << img_pts_2_[i][j].x() << ", " << img_pts_2_[i][j].y() 
                        //          << "], obj=[" << obj_pts_2_[i][j].x() << ", " << obj_pts_2_[i][j].y() << ", " << obj_pts_2_[i][j].z()
                        //          << "], res=[" << res[0] << ", " << res[1] << "], res_norm=" << std::sqrt(n2) << std::endl;
                    }

                    // out << "cam2," << i << "," << j << ","
                    //     << res[0] << "," << res[1] << "," << std::sqrt(n2) << "\n";
                }
                if (debug_frame) {
                    double rms = (cam_counts[2] > 0) ? std::sqrt(cam_err_sq[2] / cam_counts[2]) : 0.0;
                    std::cout << "CAM2: " << cam_counts[2] << " points, Sum squared error: " << cam_err_sq[2] << ", RMS: " << rms << std::endl;
                }
            } else if (debug_frame) {
                std::cout << "\n--- CAM2: No observations ---" << std::endl;
            }

            // Store RMS for each camera that observed the frame
            for (int cam = 0; cam < 3; ++cam) {
                double rms = (cam_counts[cam] > 0) ?
                            std::sqrt(cam_err_sq[cam] / cam_counts[cam]) :
                            0.0;
                frame_errors[idx * 3 + cam] = rms;
            }
            
            if (debug_frame) {
                std::cout << "Final frame errors: cam0=" << frame_errors[idx * 3 + 0] 
                         << ", cam1=" << frame_errors[idx * 3 + 1] 
                         << ", cam2=" << frame_errors[idx * 3 + 2] << std::endl;
                std::cout << "========== END CALIB DEBUG ==========\n" << std::endl;
            }
        }

        double rms = std::sqrt(total_sq_err / total_pts);
        // std::cout << "Iteration " << summary.iteration
        //         << " RMS reprojection error: " << rms << " px" << std::endl;

        std::string filename = log_dir_ + "/calib_iter_" +
            std::to_string(summary.iteration) + ".json";

        SaveCalibrationResult(
            filename,
            intrinsic_0_, dist_0_,
            intrinsic_1_, dist_1_,
            intrinsic_2_, dist_2_,
            qvec_cam_1_, tvec_cam_1_,
            qvec_cam_2_, tvec_cam_2_,
            target_poses_,
            master_timestamps_,
            frame_errors      // <--- new data!
        );

        return ceres::SOLVER_CONTINUE;
    }


private:
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0_;
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0_;
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1_;
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1_;
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_2_;
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_2_;
    const std::vector<TimestampEntry>& master_timestamps_;
    std::vector<std::array<double,7>>& target_poses_;
    double* intrinsic_0_; double* dist_0_;
    double* intrinsic_1_; double* dist_1_;
    double* intrinsic_2_; double* dist_2_;
    double* qvec_cam_1_; double* tvec_cam_1_;
    double* qvec_cam_2_; double* tvec_cam_2_;
    std::string log_dir_;
};





void OptimizeFishEyeParameters(
    double intrinsic_0[4], double dist_0[4],
    // remove extrinsics_0 from being optimized; we use target_poses instead
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0,
    double intrinsic_1[4], double dist_1[4],
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1,
    double intrinsic_2[4], double dist_2[4],
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_2,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_2,
    // inter-camera transforms remain as optimizable blocks
    double qvec_cam_1[4], double tvec_cam_1[3],
    double qvec_cam_2[4], double tvec_cam_2[3],
    // NEW: single set of target poses, one per master_timestamps entry.
    // Each array is {qw, qx, qy, qz, tx, ty, tz} representing target->cam0 (X_cam0 = R(q)*X_obj + t)
    std::vector<std::array<double,7>>& target_poses,
    const std::vector<TimestampEntry>& master_timestamps,
    const OptimizationFlags& flags = OptimizationFlags(),
    bool fix_cam1_extrinsics = false,
    bool fix_cam2_extrinsics = false
)
{
    //print out all arguments
    std::cout << "Starting optimization with parameters:" << std::endl;
    // std::cout << "Intrinsic 0: " << intrinsic_0[0] << ", " << intrinsic_0[1] << ", " << intrinsic_0[2] << ", " << intrinsic_0[3] << std::endl;
    // std::cout << "Distortion 0: " << dist_0[0] << ", " << dist_0[1] << ", " << dist_0[2] << ", " << dist_0[3] << std::endl;
    // std::cout << "Intrinsic 1: " << intrinsic_1[0] << ", " << intrinsic_1[1] << ", " << intrinsic_1[2] << ", " << intrinsic_1[3] << std::endl;
    // std::cout << "Distortion 1: " << dist_1[0] << ", " << dist_1[1] << ", " << dist_1[2] << ", " << dist_1[3] << std::endl;
    // std::cout << "Intrinsic 2: " << intrinsic_2[0] << ", " << intrinsic_2[1] << ", " << intrinsic_2[2] << ", " << intrinsic_2[3] << std::endl;
    // std::cout << "Distortion 2: " << dist_2[0] << ", " << dist_2[1] << ", " << dist_2[2] << ", " << dist_2[3] << std::endl;
    // std::cout << "qvec_cam_1: " << qvec_cam_1[0] << ", " << qvec_cam_1[1] << ", " << qvec_cam_1[2] << ", " << qvec_cam_1[3] << std::endl;
    // std::cout << "tvec_cam_1: " << tvec_cam_1[0] << ", " << tvec_cam_1[1] << ", " << tvec_cam_1[2] << std::endl;
    // std::cout << "qvec_cam_2: " << qvec_cam_2[0] << ", " << qvec_cam_2[1] << ", " << qvec_cam_2[2] << ", " << qvec_cam_2[3] << std::endl;
    // std::cout << "tvec_cam_2: " << tvec_cam_2[0] << ", " << tvec_cam_2[1] << ", " << tvec_cam_2[2] << std::endl;
    // std::cout << "Number of target poses: " << target_poses.size() << std::endl;
    // std::cout << "Number of master timestamps: " << master_timestamps.size() << std::endl;
    ceres::Problem problem;

    // Bookkeeping which target_poses entries correspond to at least one observation
    std::vector<bool> target_pose_used(target_poses.size(), false);
    // output master_timestamps size
    // std::cout << "Number of master timestamps: " << master_timestamps.size() << std::endl;

    // Add reprojection residuals
    for (size_t idx = 0; idx < master_timestamps.size(); ++idx) {
        const auto& entry = master_timestamps[idx];
        // std::cout << "Processing timestamp index " << idx << std::endl;

        // CAM0: direct use of target_poses[idx]
        if (entry.cam0_idx != -1) {
            int cam0_i = entry.cam0_idx;
            target_pose_used[idx] = true;
            // std::cout << "Processing cam0 for timestamp index " << idx << std::endl;

            // For each corner observed by cam0 at that timestamp:
            for (size_t j = 0; j < img_pts_0[cam0_i].size(); ++j) {
                // std::cout << "Processing cam0, frame " << cam0_i << ", point " << j << std::endl;
                // std::cout << "img_pts_0 size: " << img_pts_0.size() << std::endl;
                // std::cout << "obj_pts_0 size: " << obj_pts_0.size() << std::endl;
                // std::cout << "img_pts_0[cam0_i] size: " << img_pts_0[cam0_i].size() << std::endl;
                auto measured = img_pts_0[cam0_i][j];
                // std::cout << "Measured point: " << measured(0) << ", " << measured(1) << std::endl;
                auto objp = obj_pts_0[cam0_i][j];
                // std::cout << "Object point: " << objp(0) << ", " << objp(1) << ", " << objp(2) << std::endl;

                // Note: for cam0 we can make cam_q = identity, cam_t = zero so that
                // transformation via cam inverse is a no-op.
                // We will pass cam_q_cam0 = (1,0,0,0) and cam_t_cam0 = (0,0,0)
                static double cam0_q[4] = {1.0, 0.0, 0.0, 0.0};
                // std::cout << "cam0_q: [" << cam0_q[0] << ", " << cam0_q[1] << ", " << cam0_q[2] << ", " << cam0_q[3] << "]" << std::endl;
                static double cam0_t[3] = {0.0, 0.0, 0.0};
                // std::cout << "cam0_q: [" << cam0_q[0] << ", " << cam0_q[1] << ", " << cam0_q[2] << ", " << cam0_q[3] << "]" << std::endl;

                // Create cost function that depends on target_pose (in cam0) and cam0 identity
                ceres::CostFunction* cost = FisheyeReproj_TargetInCam0::Create(measured, objp);

                double* target_q = target_poses[idx].data();          // qw,qx,qy,qz
                double* target_t = target_poses[idx].data() + 4;      // tx,ty,tz

                // Add residual: depends on intrinsics, dist, target_pose, and cam pose in cam0
                // For cam0, cam pose params are identity constants - we pass pointer to statics but
                // do not add them as parameter blocks (they are constant in problem.AddResidualBlock call)
                problem.AddResidualBlock(cost, nullptr,
                                         intrinsic_0, dist_0,
                                         target_q, target_t,
                                         cam0_q, cam0_t);
                // std::cout << "Added residual block for cam0, frame " << cam0_i << ", point " << j << std::endl;
                // fix intrinsics of cam0 if desired:
                if (!flags.optimize_intrinsics) {
                    problem.SetParameterBlockConstant(intrinsic_0);
                    problem.SetParameterBlockConstant(dist_0);
                }
                // fix extrinsics of cam0 (identity) - always fixed for cam 0
                problem.AddParameterBlock(cam0_q, 4);
                problem.SetParameterBlockConstant(cam0_q);
                problem.AddParameterBlock(cam0_t, 3);
                problem.SetParameterBlockConstant(cam0_t);
                // std::cout << "Added residual for cam0, frame " << cam0_i << ", point " << j << std::endl;
            }
        }

        // CAM1: use target_poses[idx] & (qvec_cam_1, tvec_cam_1)
        if (entry.cam1_idx != -1) {
            int cam1_i = entry.cam1_idx;
            target_pose_used[idx] = true;

            for (size_t j = 0; j < img_pts_1[cam1_i].size(); ++j) {
                auto measured = img_pts_1[cam1_i][j];
                auto objp = obj_pts_1[cam1_i][j];

                ceres::CostFunction* cost = FisheyeReproj_TargetInCam0::Create(measured, objp);

                double* target_q = target_poses[idx].data();
                double* target_t = target_poses[idx].data() + 4;

                problem.AddResidualBlock(cost, nullptr,
                                         intrinsic_1, dist_1,
                                         target_q, target_t,
                                         qvec_cam_1, tvec_cam_1);
                // fix intrinsics of cam1 if desired:
                if (!flags.optimize_intrinsics) {
                    problem.SetParameterBlockConstant(intrinsic_1);
                    problem.SetParameterBlockConstant(dist_1);
                }
            }
        }

        // CAM2: use target_poses[idx] & (qvec_cam_2, tvec_cam_2)
        if (entry.cam2_idx != -1) {
            int cam2_i = entry.cam2_idx;
            target_pose_used[idx] = true;

            for (size_t j = 0; j < img_pts_2[cam2_i].size(); ++j) {
                auto measured = img_pts_2[cam2_i][j];
                auto objp = obj_pts_2[cam2_i][j];

                ceres::CostFunction* cost = FisheyeReproj_TargetInCam0::Create(measured, objp);

                double* target_q = target_poses[idx].data();
                double* target_t = target_poses[idx].data() + 4;

                problem.AddResidualBlock(cost, nullptr,
                                         intrinsic_2, dist_2,
                                         target_q, target_t,
                                         qvec_cam_2, tvec_cam_2);
                // fix intrinsics of cam2 if desired:
                if (!flags.optimize_intrinsics) {
                    problem.SetParameterBlockConstant(intrinsic_2);
                    problem.SetParameterBlockConstant(dist_2);
                }
            }
        }
    }

    std::cout << "Total target poses used in observations: "
              << std::count(target_pose_used.begin(), target_pose_used.end(), true)
              << " out of " << target_poses.size() << std::endl;

    for (size_t i = 0; i < target_poses.size(); ++i) {
        if (!target_pose_used[i]) continue;
        problem.AddParameterBlock(target_poses[i].data(), 4, new ceres::QuaternionManifold());
        problem.AddParameterBlock(target_poses[i].data() + 4, 3);
    }

    problem.AddParameterBlock(qvec_cam_1, 4, new ceres::QuaternionManifold());
    problem.AddParameterBlock(tvec_cam_1, 3);
    problem.AddParameterBlock(qvec_cam_2, 4, new ceres::QuaternionManifold());
    problem.AddParameterBlock(tvec_cam_2, 3);
    // Optionally fix inter-camera extrinsics
    if (!flags.optimize_inter_camera) {
        problem.SetParameterBlockConstant(qvec_cam_1);
        problem.SetParameterBlockConstant(tvec_cam_1);
        problem.SetParameterBlockConstant(qvec_cam_2);
        problem.SetParameterBlockConstant(tvec_cam_2);
    } else {
        // Selectively fix specific extrinsics if requested
        if (fix_cam1_extrinsics) {
            problem.SetParameterBlockConstant(qvec_cam_1);
            problem.SetParameterBlockConstant(tvec_cam_1);
        }
        if (fix_cam2_extrinsics) {
            problem.SetParameterBlockConstant(qvec_cam_2);
            problem.SetParameterBlockConstant(tvec_cam_2);
        }
    }

    // Normalize initial quaternions for target_poses and cam quaternions
    for (auto &tp : target_poses) {
        Eigen::Map<Eigen::Quaterniond> q(tp.data());
        q.normalize();
    }
    {
        Eigen::Map<Eigen::Quaterniond> q(qvec_cam_1); q.normalize();
    }
    {
        Eigen::Map<Eigen::Quaterniond> q(qvec_cam_2); q.normalize();
    }
    std::cout << "Starting Ceres Solver..." << std::endl;

    // Solver options
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    // tune as you like (max_num_iterations, trust_region settings...)
    options.update_state_every_iteration = true;
    options.max_num_iterations = 50;
    // options.initial_trust_region_radius = 0.001;

// Add your save callback
    // options.callbacks.push_back(
    //     new SaveIterationCallback(
    //         intrinsic_0, dist_0,
    //         intrinsic_1, dist_1,
    //         intrinsic_2, dist_2,
    //         qvec_cam_1, tvec_cam_1,
    //         qvec_cam_2, tvec_cam_2,
    //         &target_poses, &master_timestamps,
    //         "/home/jake/calibration_w_eigen",
    //         &problem
    //     )
    // );
    options.callbacks.push_back(new ResidualEvalCallback(
        img_pts_0, obj_pts_0,
        img_pts_1, obj_pts_1,
        img_pts_2, obj_pts_2,
        master_timestamps,
        &target_poses,
        intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        "/home/jake/calibration_w_eigen"
    ));


    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << summary.BriefReport() << std::endl;
}


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






void PrintMasterTimestamps(const std::vector<TimestampEntry>& master_timestamps,
                           const std::vector<size_t>& filtered_timestamp_list_0,
                           const std::vector<size_t>& filtered_timestamp_list_1,
                           const std::vector<size_t>& filtered_timestamp_list_2)
{
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "Master Timestamp Alignment:\n";
    std::cout << "Index |   Timestamp   | Cam0_idx | Cam0_time | Cam1_idx | Cam1_time | Cam2_idx | Cam2_time\n";
    std::cout << "---------------------------------------------------------------\n";

    for (size_t i = 0; i < master_timestamps.size(); ++i) {
        const auto& entry = master_timestamps[i];

        std::cout << std::setw(5) << i << " | "
                  << std::setw(13) << entry.timestamp_id << " | "
                  << std::setw(8)  << entry.cam0_idx << " | ";

        if (entry.cam0_idx != -1)
            std::cout << std::setw(10) << filtered_timestamp_list_0[entry.cam0_idx] << " | ";
        else
            std::cout << "     ---    | ";

        std::cout << std::setw(8)  << entry.cam1_idx << " | ";

        if (entry.cam1_idx != -1)
            std::cout << std::setw(10) << filtered_timestamp_list_1[entry.cam1_idx] << " | ";
        else
            std::cout << "     ---    | ";

        std::cout << std::setw(8)  << entry.cam2_idx << " | ";

        if (entry.cam2_idx != -1)
            std::cout << std::setw(10) << filtered_timestamp_list_2[entry.cam2_idx];
        else
            std::cout << "     ---    ";

        std::cout << std::endl;
    }

    std::cout << "---------------------------------------------------------------\n";
}

// Convert quaternion+translation (w,x,y,z, tx,ty,tz) to 4x4 transform (target in cam frame)
Eigen::Matrix4d quatTransToMatrixLoaded(const std::array<double,7>& a) {
    double w = a[0], x = a[1], y = a[2], z = a[3];
    // normalize
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm <= 1e-12) { w = 1; x=y=z=0; norm=1.0; }
    w/=norm; x/=norm; y/=norm; z/=norm;

    Eigen::Matrix3d R;
    // quaternion to rotation matrix (w, x, y, z)
    R(0,0) = 1 - 2*(y*y + z*z);
    R(0,1) = 2*(x*y - z*w);
    R(0,2) = 2*(x*z + y*w);
    R(1,0) = 2*(x*y + z*w);
    R(1,1) = 1 - 2*(x*x + z*z);
    R(1,2) = 2*(y*z - x*w);
    R(2,0) = 2*(x*z - y*w);
    R(2,1) = 2*(y*z + x*w);
    R(2,2) = 1 - 2*(x*x + y*y);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T(0,3) = a[4];
    T(1,3) = a[5];
    T(2,3) = a[6];
    return T;
}

// Load function: returns vector of {w,x,y,z,tx,ty,tz}
bool LoadTargetPosesFromJson(const std::string& filepath,
                             std::vector<std::array<double,7>>& out_target_poses,
                             std::vector<double>* out_timestamps = nullptr) {
    std::ifstream ifs(filepath);
    if (!ifs.good()) {
        std::cerr << "Failed to open " << filepath << std::endl;
        return false;
    }
    json j;
    ifs >> j;

    if (!j.contains("target_poses") || !j["target_poses"].is_array()) {
        std::cerr << "JSON does not contain target_poses array\n";
        return false;
    }

    const auto& arr = j["target_poses"];
    out_target_poses.clear();
    out_target_poses.reserve(arr.size());

    if (out_timestamps) out_timestamps->clear();

    for (const auto& e : arr) {
        // Expect "quaternion": [w,x,y,z], "translation": [tx,ty,tz], optional "timestamp"
        if (!e.contains("quaternion") || !e.contains("translation")) {
            std::cerr << "target_poses entry missing quaternion or translation\n";
            return false;
        }

        auto q = e["quaternion"];
        auto t = e["translation"];
        if (q.size() != 4 || t.size() != 3) {
            std::cerr << "Invalid quaternion/translation size\n";
            return false;
        }

        std::array<double,7> a;
        a[0] = q[0].get<double>(); // w
        a[1] = q[1].get<double>(); // x
        a[2] = q[2].get<double>(); // y
        a[3] = q[3].get<double>(); // z
        a[4] = t[0].get<double>();
        a[5] = t[1].get<double>();
        a[6] = t[2].get<double>();

        // normalize quaternion
        double n = std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
        if (n < 1e-12) { a[0]=1; a[1]=a[2]=a[3]=0; }
        else for (int k=0;k<4;++k) a[k] /= n;

        out_target_poses.push_back(a);

        if (out_timestamps) {
            if (e.contains("timestamp"))
                out_timestamps->push_back(e["timestamp"].get<double>());
            else
                out_timestamps->push_back(0.0);
        }
    }

    return true;
}



int main(int argc, char** argv) {
    // Parse command-line arguments
    std::string data_file;
    std::string target_poses_file;
    std::string intrinsics_file;
    std::string extrinsics_file;
    std::string per_frame_flags_file;
    std::string global_flags_file;
    
    // Simple argument parser
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-datafile") {
            if (i + 1 < argc) {
                data_file = argv[++i];
            } else {
                std::cerr << "Error: -datafile requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "-intrinsicsfile") {
            if (i + 1 < argc) {
                intrinsics_file = argv[++i];
            } else {
                std::cerr << "Error: -intrinsicsfile requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "-extrinsicsfile") {
            if (i + 1 < argc) {
                extrinsics_file = argv[++i];
            } else {
                std::cerr << "Error: -extrinsicsfile requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "-calibrationfile") {
            if (i + 1 < argc) {
                target_poses_file = argv[++i];
            } else {
                std::cerr << "Error: -calibrationfile requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "-perframeflags") {
            if (i + 1 < argc) {
                per_frame_flags_file = argv[++i];
            } else {
                std::cerr << "Error: -perframeflags requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "-globalflags") {
            if (i + 1 < argc) {
                global_flags_file = argv[++i];
            } else {
                std::cerr << "Error: -globalflags requires a file path" << std::endl;
                return -1;
            }
        } else if (arg == "--help" || arg == "-h" || arg == "-help") {
            std::cout << "Usage: " << argv[0] << " -datafile <data_file.csv> [options]\n"
                      << "Required:\n"
                      << "  -datafile <file>              CSV data file with corner detections\n"
                      << "Options:\n"
                      << "  -intrinsicsfile <file>        (optional) Load initial intrinsics from JSON file\n"
                      << "  -extrinsicsfile <file>        (optional) Load inter-camera extrinsics from JSON file\n"
                      << "  -calibrationfile <file>       (optional) Load target poses from JSON file\n"
                      << "  -perframeflags <file>         (optional) Load per-frame optimization flags from JSON file\n"
                      << "  -globalflags <file>            (optional) Load global optimization flags from JSON file\n"
                      << "  -h, --help                    Show this help message\n"
                      << "\n"
                      << "Examples:\n"
                      << "  " << argv[0] << " -datafile data.csv\n"
                      << "  " << argv[0] << " -datafile data.csv -intrinsicsfile intrinsics.json\n"
                      << "  " << argv[0] << " -datafile data.csv -extrinsicsfile extrinsics.json\n"
                      << "  " << argv[0] << " -datafile data.csv -calibrationfile poses.json -intrinsicsfile intrinsics.json -extrinsicsfile extrinsics.json\n"
                      << "  " << argv[0] << " -datafile data.csv -perframeflags per_frame_flags.json -globalflags global_flags.json\n";
            return 0;
        } else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            std::cerr << "Use -h or --help for usage information" << std::endl;
            return -1;
        }
    }
    
    // Check required arguments
    if (data_file.empty()) {
        std::cerr << "Error: Missing required argument: -datafile\n"
                  << "Usage: " << argv[0] << " -datafile <data_file.csv> [options]\n"
                  << "Use -h or --help for more information" << std::endl;
        return -1;
    }
    
    // Note: target_poses_file is optional - if not provided, Zhang's method will be used
    // (when the JSON loading code is commented out)

    // Step 1: Load and process CSV data for all cameras
    auto [obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0] = processCSV(data_file, 0);
    auto [obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1] = processCSV(data_file, 1);
    auto [obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2] = processCSV(data_file, 2);

    // ========== ZHANG'S METHOD INITIALIZATION (COMMENTED OUT) ==========
    // Step 2: Compute homographies and filter timestamps
    // auto [H_list_0, filtered_timestamp_list_0] = computeHomographies(obj_pts_list_0, img_pts_list_0, timestamp_list_0);
    // auto [H_list_1, filtered_timestamp_list_1] = computeHomographies(obj_pts_list_1, img_pts_list_1, timestamp_list_1);
    // auto [H_list_2, filtered_timestamp_list_2] = computeHomographies(obj_pts_list_2, img_pts_list_2, timestamp_list_2);

    // Filter data for all cameras
    // std::tie(obj_pts_list_0, img_pts_list_0, corner_ids_list_0) = filterDataByTimestamps(
    //     obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0, filtered_timestamp_list_0);
    // std::tie(obj_pts_list_1, img_pts_list_1, corner_ids_list_1) = filterDataByTimestamps(
    //     obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1, filtered_timestamp_list_1);
    // std::tie(obj_pts_list_2, img_pts_list_2, corner_ids_list_2) = filterDataByTimestamps(
    //     obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2, filtered_timestamp_list_2);
    // ========== END ZHANG'S METHOD INITIALIZATION ==========

    // NEW INITIALIZATION: Use all timestamps (no filtering by homography)
    // Create filtered timestamp lists from all available timestamps
    std::vector<size_t> filtered_timestamp_list_0, filtered_timestamp_list_1, filtered_timestamp_list_2;
    filtered_timestamp_list_0 = timestamp_list_0;
    filtered_timestamp_list_1 = timestamp_list_1;
    filtered_timestamp_list_2 = timestamp_list_2;


    auto buildIndexMap = [](const std::vector<size_t>& timestamps) -> std::unordered_map<size_t, int> {
        std::unordered_map<size_t, int> map;
        for (int i = 0; i < timestamps.size(); ++i) {
            map[timestamps[i]] = i;
        }
        return map;
    };
    
    auto map0 = buildIndexMap(filtered_timestamp_list_0);
    auto map1 = buildIndexMap(filtered_timestamp_list_1);
    auto map2 = buildIndexMap(filtered_timestamp_list_2);
    
    // Union of all timestamps
    std::set<size_t> all_timestamps;
    all_timestamps.insert(filtered_timestamp_list_0.begin(), filtered_timestamp_list_0.end());
    all_timestamps.insert(filtered_timestamp_list_1.begin(), filtered_timestamp_list_1.end());
    all_timestamps.insert(filtered_timestamp_list_2.begin(), filtered_timestamp_list_2.end());
    
    std::vector<TimestampEntry> master_timestamps;
    for (auto t : all_timestamps) {
        master_timestamps.push_back({
            t,
            map0.count(t) ? map0[t] : -1,
            map1.count(t) ? map1[t] : -1,
            map2.count(t) ? map2[t] : -1
        });
    }

    PrintMasterTimestamps(master_timestamps,
                      filtered_timestamp_list_0,
                      filtered_timestamp_list_1,
                      filtered_timestamp_list_2);

    // std::cin.get();  // Wait for user input before proceeding


    // ========== ZHANG'S METHOD INITIALIZATION (COMMENTED OUT) ==========
    // Initialize camera parameters
    // double intrinsic_0[4];
    // double dist_0[4];
    // double intrinsic_1[4];
    // double dist_1[4];
    // double intrinsic_2[4];
    // double dist_2[4];
    // 
    // // Load intrinsics from file if provided, otherwise use defaults
    // bool intrinsics_loaded = false;
    // if (!intrinsics_file.empty()) {
    //     intrinsics_loaded = LoadIntrinsicsFromJson(intrinsics_file,
    //                                                 intrinsic_0, dist_0,
    //                                                 intrinsic_1, dist_1,
    //                                                 intrinsic_2, dist_2);
    // }
    // 
    // // If not loaded from file, estimate from homographies using robust_intrinsic_estimation
    // if (!intrinsics_loaded) {
    //     std::cout << "Estimating intrinsics from homographies using robust_intrinsic_estimation..." << std::endl;
    //     
    //     // Estimate intrinsics for each camera from their homographies
    //     Eigen::Matrix3d K_0_est = robust_intrinsic_estimation(H_list_0);
    //     Eigen::Matrix3d K_1_est = robust_intrinsic_estimation(H_list_1);
    //     Eigen::Matrix3d K_2_est = robust_intrinsic_estimation(H_list_2);
    //     
    //     // Check if estimation succeeded (not identity matrix)
    //     bool estimation_succeeded = true;
    //     if (K_0_est.isApprox(Eigen::Matrix3d::Identity()) || 
    //         K_1_est.isApprox(Eigen::Matrix3d::Identity()) || 
    //         K_2_est.isApprox(Eigen::Matrix3d::Identity())) {
    //         estimation_succeeded = false;
    //     }
    //     
    //     if (estimation_succeeded) {
    //         // Extract intrinsics from estimated K matrices
    //         intrinsic_0[0] = K_0_est(0, 0); intrinsic_0[1] = K_0_est(1, 1); intrinsic_0[2] = K_0_est(0, 2); intrinsic_0[3] = K_0_est(1, 2);
    //         intrinsic_1[0] = K_1_est(0, 0); intrinsic_1[1] = K_1_est(1, 1); intrinsic_1[2] = K_1_est(0, 2); intrinsic_1[3] = K_1_est(1, 2);
    //         intrinsic_2[0] = K_2_est(0, 0); intrinsic_2[1] = K_2_est(1, 1); intrinsic_2[2] = K_2_est(0, 2); intrinsic_2[3] = K_2_est(1, 2);
    //         
    //         // Initialize distortion coefficients to small values (will be optimized)
    //         dist_0[0] = -0.04; dist_0[1] = 0.03; dist_0[2] = -0.04; dist_0[3] = 0.015;
    //         dist_1[0] = -0.04; dist_1[1] = 0.03; dist_1[2] = -0.04; dist_1[3] = 0.015;
    //         dist_2[0] = -0.04; dist_2[1] = 0.03; dist_2[2] = -0.04; dist_2[3] = 0.015;
    //         
    //         std::cout << "Successfully estimated intrinsics from homographies." << std::endl;
    //         std::cout << "Camera 0: fx=" << intrinsic_0[0] << ", fy=" << intrinsic_0[1] 
    //                   << ", cx=" << intrinsic_0[2] << ", cy=" << intrinsic_0[3] << std::endl;
    //         std::cout << "Camera 1: fx=" << intrinsic_1[0] << ", fy=" << intrinsic_1[1] 
    //                   << ", cx=" << intrinsic_1[2] << ", cy=" << intrinsic_1[3] << std::endl;
    //         std::cout << "Camera 2: fx=" << intrinsic_2[0] << ", fy=" << intrinsic_2[1] 
    //                   << ", cx=" << intrinsic_2[2] << ", cy=" << intrinsic_2[3] << std::endl;
    //     } else {
    //         // Fallback to default intrinsics if estimation failed
    //         std::cout << "Warning: Intrinsic estimation failed, using default values." << std::endl;
    //         Eigen::Matrix3d K_0_default;
    //         K_0_default << 800, 0, 640,
    //                        0, 800, 480,
    //                        0, 0, 1;
    //         Eigen::Matrix3d K_1_default;
    //         K_1_default << 800, 0, 640,
    //                        0, 800, 480,
    //                        0, 0, 1;
    //         Eigen::Matrix3d K_2_default;
    //         K_2_default << 800, 0, 640,
    //                        0, 800, 480,
    //                        0, 0, 1;
    //         intrinsic_0[0] = K_0_default(0, 0); intrinsic_0[1] = K_0_default(1, 1); intrinsic_0[2] = K_0_default(0, 2); intrinsic_0[3] = K_0_default(1, 2);
    //         dist_0[0] = -0.04; dist_0[1] = 0.03; dist_0[2] = -0.04; dist_0[3] = 0.015;
    //         intrinsic_1[0] = K_1_default(0, 0); intrinsic_1[1] = K_1_default(1, 1); intrinsic_1[2] = K_1_default(0, 2); intrinsic_1[3] = K_1_default(1, 2);
    //         dist_1[0] = -0.04; dist_1[1] = 0.03; dist_1[2] = -0.04; dist_1[3] = 0.015;
    //         intrinsic_2[0] = K_2_default(0, 0); intrinsic_2[1] = K_2_default(1, 1); intrinsic_2[2] = K_2_default(0, 2); intrinsic_2[3] = K_2_default(1, 2);
    //         dist_2[0] = -0.04; dist_2[1] = 0.03; dist_2[2] = -0.04; dist_2[3] = 0.015;
    //     }
    // }
    // 
    // // Reconstruct K matrices from intrinsics (needed for compute_extrinsic_params)
    // Eigen::Matrix3d K_0, K_1, K_2;
    // K_0 << intrinsic_0[0], 0.0, intrinsic_0[2],
    //         0.0, intrinsic_0[1], intrinsic_0[3],
    //         0.0, 0.0, 1.0;
    // K_1 << intrinsic_1[0], 0.0, intrinsic_1[2],
    //         0.0, intrinsic_1[1], intrinsic_1[3],
    //         0.0, 0.0, 1.0;
    // K_2 << intrinsic_2[0], 0.0, intrinsic_2[2],
    //         0.0, intrinsic_2[1], intrinsic_2[3],
    //         0.0, 0.0, 1.0;
    // 
    // // Add extrinsics for all cameras
    // std::vector<std::array<double, 7>> extrinsics_0, extrinsics_1, extrinsics_2;
    // for (const auto& H : H_list_0) {
    //     auto [R, t] = compute_extrinsic_params(H, K_0);
    //     Eigen::Quaterniond q(R);
    //     std::array<double, 7> pose;
    //     pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
    //     pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
    //     extrinsics_0.push_back(pose);
    // }
    // for (const auto& H : H_list_1) {
    //     auto [R, t] = compute_extrinsic_params(H, K_1);
    //     Eigen::Quaterniond q(R);
    //     std::array<double, 7> pose;
    //     pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
    //     pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
    //     extrinsics_1.push_back(pose);
    // }
    // for (const auto& H : H_list_2) {
    //     auto [R, t] = compute_extrinsic_params(H, K_2);
    //     Eigen::Quaterniond q(R);
    //     std::array<double, 7> pose;
    //     pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
    //     pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
    //     extrinsics_2.push_back(pose);
    // }
    // ========== END ZHANG'S METHOD INITIALIZATION ==========

    // ========== NEW INITIALIZATION SCHEMA ==========
    // Initialize camera parameters
    double intrinsic_0[4];
    double dist_0[4];
    double intrinsic_1[4];
    double dist_1[4];
    double intrinsic_2[4];
    double dist_2[4];
    
    // Load initial intrinsics from JSON file (required for new initialization)
    if (intrinsics_file.empty()) {
        std::cerr << "Error: New initialization schema requires -intrinsicsfile to be provided." << std::endl;
        return -1;
    }
    
    bool intrinsics_loaded = LoadIntrinsicsFromJson(intrinsics_file,
                                                    intrinsic_0, dist_0,
                                                    intrinsic_1, dist_1,
                                                    intrinsic_2, dist_2);
    if (!intrinsics_loaded) {
        std::cerr << "Error: Failed to load intrinsics from file. Required for new initialization schema." << std::endl;
        return -1;
    }
    
    std::cout << "Loaded initial intrinsics from JSON file." << std::endl;
    
    // Helper functions for pose averaging
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
        Eigen::Quaterniond q(T.block<3,3>(0,0));
        std::array<double,7> a;
        a[0]=q.w(); a[1]=q.x(); a[2]=q.y(); a[3]=q.z();
        a[4]=T(0,3); a[5]=T(1,3); a[6]=T(2,3);
        return a;
    };
    
    // Log/exp helpers for averaging rotations
    auto logSO3 = [](const Eigen::Matrix3d& R) {
        Eigen::AngleAxisd aa(R);
        return aa.angle() * aa.axis();
    };
    auto expSO3 = [](const Eigen::Vector3d& r) -> Eigen::Matrix3d {
        double theta = r.norm();
        if (theta < 1e-12) return Eigen::Matrix3d::Identity();
        Eigen::AngleAxisd aa(theta, r.normalized());
        return aa.toRotationMatrix();
    };
    
    // Average SE(3) transforms
    auto averagePoses = [&](const std::vector<Eigen::Matrix4d>& Ts) -> Eigen::Matrix4d {
        if (Ts.empty()) {
            Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
            return result;
        }
        Eigen::Vector3d t_avg = Eigen::Vector3d::Zero();
        Eigen::Vector3d r_accum = Eigen::Vector3d::Zero();
        for (const auto& T : Ts) {
            t_avg += T.block<3,1>(0,3);
            r_accum += logSO3(T.block<3,3>(0,0));
        }
        t_avg /= Ts.size();
        r_accum /= Ts.size();
        Eigen::Matrix4d T_avg = Eigen::Matrix4d::Identity();
        T_avg.block<3,3>(0,0) = expSO3(r_accum);
        T_avg.block<3,1>(0,3) = t_avg;
        return T_avg;
    };
    
    // Helper: Initialize target pose in front of camera so all object points are visible
    auto initializeTargetPoseInFrontOfCamera = [](const std::vector<Eigen::Vector3d>& obj_pts) -> std::array<double,7> {
        if (obj_pts.empty()) {
            return {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5}; // Default: 0.5m in front
        }
        
        // Compute bounding box of object points
        double min_x = obj_pts[0].x(), max_x = obj_pts[0].x();
        double min_y = obj_pts[0].y(), max_y = obj_pts[0].y();
        for (const auto& pt : obj_pts) {
            min_x = std::min(min_x, pt.x());
            max_x = std::max(max_x, pt.x());
            min_y = std::min(min_y, pt.y());
            max_y = std::max(max_y, pt.y());
        }
        
        // Center of the board
        double center_x = (min_x + max_x) / 2.0;
        double center_y = (min_y + max_y) / 2.0;
        
        // Distance to place target: ensure all points are visible
        // Use a reasonable distance (e.g., 0.5m) with target centered at origin in camera frame
        // Target pose: identity rotation, translation to center the board in front of camera
        std::array<double,7> pose;
        pose[0] = 1.0; pose[1] = 0.0; pose[2] = 0.0; pose[3] = 0.0; // Identity quaternion
        pose[4] = -center_x; // Center X
        pose[5] = -center_y; // Center Y
        pose[6] = 0.5; // 0.5m in front along Z
        
        return pose;
    };
    
    // Note: master_timestamps is already built above (lines 2145-2176), no need to rebuild it here
    
    // Storage for per-frame, per-camera optimized intrinsics and target poses
    std::vector<std::array<double, 4>> per_frame_intrinsics_0, per_frame_intrinsics_1, per_frame_intrinsics_2;
    std::vector<std::array<double, 4>> per_frame_dist_0, per_frame_dist_1, per_frame_dist_2;
    // Per-frame, per-camera target poses (in each camera's frame)
    std::vector<std::vector<std::array<double,7>>> per_frame_target_poses_by_cam;
    per_frame_target_poses_by_cam.resize(master_timestamps.size());
    
    std::cout << "\n========== NEW INITIALIZATION: Per-frame, per-camera optimization ==========" << std::endl;
    
    // Step 2: For each frame, for each camera with data, optimize intrinsics and target pose
    for (size_t frame_idx = 0; frame_idx < master_timestamps.size(); ++frame_idx) {
        const auto& entry = master_timestamps[frame_idx];
        std::cout << "\nProcessing frame " << frame_idx << " (timestamp " << entry.timestamp_id << ")..." << std::endl;
        
        // Track optimized intrinsics for this frame (will be updated as cameras are optimized)
        double frame_intrinsic_0[4] = {intrinsic_0[0], intrinsic_0[1], intrinsic_0[2], intrinsic_0[3]};
        double frame_dist_0[4] = {dist_0[0], dist_0[1], dist_0[2], dist_0[3]};
        double frame_intrinsic_1[4] = {intrinsic_1[0], intrinsic_1[1], intrinsic_1[2], intrinsic_1[3]};
        double frame_dist_1[4] = {dist_1[0], dist_1[1], dist_1[2], dist_1[3]};
        double frame_intrinsic_2[4] = {intrinsic_2[0], intrinsic_2[1], intrinsic_2[2], intrinsic_2[3]};
        double frame_dist_2[4] = {dist_2[0], dist_2[1], dist_2[2], dist_2[3]};
        
        // Process each camera that has data for this frame
        for (int cam_id = 0; cam_id < 3; ++cam_id) {
            int cam_idx = -1;
            const std::vector<std::vector<Eigen::Vector2d>>* img_pts_list = nullptr;
            const std::vector<std::vector<Eigen::Vector3d>>* obj_pts_list = nullptr;
            double* cam_intrinsic = nullptr;
            double* cam_dist = nullptr;
            
            if (cam_id == 0 && entry.cam0_idx != -1) {
                cam_idx = entry.cam0_idx;
                img_pts_list = &img_pts_list_0;
                obj_pts_list = &obj_pts_list_0;
                cam_intrinsic = intrinsic_0;
                cam_dist = dist_0;
            } else if (cam_id == 1 && entry.cam1_idx != -1) {
                cam_idx = entry.cam1_idx;
                img_pts_list = &img_pts_list_1;
                obj_pts_list = &obj_pts_list_1;
                cam_intrinsic = intrinsic_1;
                cam_dist = dist_1;
            } else if (cam_id == 2 && entry.cam2_idx != -1) {
                cam_idx = entry.cam2_idx;
                img_pts_list = &img_pts_list_2;
                obj_pts_list = &obj_pts_list_2;
                cam_intrinsic = intrinsic_2;
                cam_dist = dist_2;
            } else {
                continue; // This camera doesn't have data for this frame
            }
            
            std::cout << "  Optimizing camera " << cam_id << " (frame index " << cam_idx << ")..." << std::endl;
            
            // 2a) Fix camera at origin (cam0 is identity, cam1/cam2 will be handled later)
            // 2b) Initialize target pose in front of camera
            const auto& obj_pts = (*obj_pts_list)[cam_idx];
            std::array<double,7> target_pose_cam_frame = initializeTargetPoseInFrontOfCamera(obj_pts);
            
            // Create single-frame, single-camera data structures
            std::vector<std::vector<Eigen::Vector2d>> single_frame_img;
            std::vector<std::vector<Eigen::Vector3d>> single_frame_obj;
            single_frame_img.push_back((*img_pts_list)[cam_idx]);
            single_frame_obj.push_back((*obj_pts_list)[cam_idx]);
            
            std::vector<TimestampEntry> single_timestamp;
            TimestampEntry single_entry;
            single_entry.timestamp_id = entry.timestamp_id;
            single_entry.cam0_idx = (cam_id == 0) ? 0 : -1;
            single_entry.cam1_idx = (cam_id == 1) ? 0 : -1;
            single_entry.cam2_idx = (cam_id == 2) ? 0 : -1;
            single_timestamp.push_back(single_entry);
            
            std::vector<std::array<double,7>> single_target_pose = {target_pose_cam_frame};
            
            // Create local copies of intrinsics for this camera
            double local_intrinsic[4], local_dist[4];
            std::copy(cam_intrinsic, cam_intrinsic + 4, local_intrinsic);
            std::copy(cam_dist, cam_dist + 4, local_dist);
            
            // Dummy inter-camera extrinsics (not used for single-camera optimization)
            double dummy_qvec[4] = {1.0, 0.0, 0.0, 0.0};
            double dummy_tvec[3] = {0.0, 0.0, 0.0};
            
            // Save calibration results BEFORE optimization
            {
                // Prepare intrinsics arrays for save (use current frame intrinsics)
                double save_intrinsic_0[4] = {frame_intrinsic_0[0], frame_intrinsic_0[1], frame_intrinsic_0[2], frame_intrinsic_0[3]};
                double save_dist_0[4] = {frame_dist_0[0], frame_dist_0[1], frame_dist_0[2], frame_dist_0[3]};
                double save_intrinsic_1[4] = {frame_intrinsic_1[0], frame_intrinsic_1[1], frame_intrinsic_1[2], frame_intrinsic_1[3]};
                double save_dist_1[4] = {frame_dist_1[0], frame_dist_1[1], frame_dist_1[2], frame_dist_1[3]};
                double save_intrinsic_2[4] = {frame_intrinsic_2[0], frame_intrinsic_2[1], frame_intrinsic_2[2], frame_intrinsic_2[3]};
                double save_dist_2[4] = {frame_dist_2[0], frame_dist_2[1], frame_dist_2[2], frame_dist_2[3]};
                
                // Update the camera being optimized with current local values
                if (cam_id == 0) {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_0);
                    std::copy(local_dist, local_dist + 4, save_dist_0);
                } else if (cam_id == 1) {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_1);
                    std::copy(local_dist, local_dist + 4, save_dist_1);
                } else {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_2);
                    std::copy(local_dist, local_dist + 4, save_dist_2);
                }
                
                // Use initialized pose for this camera (single_timestamp has one entry, so one target pose)
                std::vector<std::array<double, 7>> frame_target_poses_before = {target_pose_cam_frame};
                
                std::string filename_before = "calibration_result_timestamp_" + std::to_string(entry.timestamp_id) + "_cam" + std::to_string(cam_id) + "_before.json";
                SaveCalibrationResult(
                    filename_before,
                    save_intrinsic_0, save_dist_0,
                    save_intrinsic_1, save_dist_1,
                    save_intrinsic_2, save_dist_2,
                    dummy_qvec, dummy_tvec, // cam1 (dummy)
                    dummy_qvec, dummy_tvec, // cam2 (dummy)
                    frame_target_poses_before,
                    single_timestamp
                );
                std::cout << "    Saved calibration results BEFORE optimization to " << filename_before << std::endl;
            }
            
            // 2d) Optimize intrinsics and target pose for this single camera+frame
            OptimizationFlags single_cam_flags;
            single_cam_flags.optimize_intrinsics = true;
            single_cam_flags.optimize_distortion = true;
            single_cam_flags.optimize_inter_camera = false; // Not optimizing inter-camera for single camera
            single_cam_flags.optimize_target_poses = true;
            
            // Create empty data for other cameras
            std::vector<std::vector<Eigen::Vector2d>> empty_img;
            std::vector<std::vector<Eigen::Vector3d>> empty_obj;
            
            if (cam_id == 0) {
                OptimizeFishEyeParameters(
                    local_intrinsic, local_dist,
                    single_frame_img, single_frame_obj,
                    intrinsic_1, dist_1, // These won't be used but need to be passed
                    empty_img, empty_obj,
                    intrinsic_2, dist_2,
                    empty_img, empty_obj,
                    dummy_qvec, dummy_tvec, // cam1 (not used)
                    dummy_qvec, dummy_tvec, // cam2 (not used)
                    single_target_pose,
                    single_timestamp,
                    single_cam_flags,
                    true, // fix cam1
                    true  // fix cam2
                );
            } else if (cam_id == 1) {
                OptimizeFishEyeParameters(
                    intrinsic_0, dist_0, // These won't be used but need to be passed
                    empty_img, empty_obj,
                    local_intrinsic, local_dist,
                    single_frame_img, single_frame_obj,
                    intrinsic_2, dist_2,
                    empty_img, empty_obj,
                    dummy_qvec, dummy_tvec, // cam1 (not used)
                    dummy_qvec, dummy_tvec, // cam2 (not used)
                    single_target_pose,
                    single_timestamp,
                    single_cam_flags,
                    true, // fix cam1
                    true  // fix cam2
                );
            } else { // cam_id == 2
                OptimizeFishEyeParameters(
                    intrinsic_0, dist_0, // These won't be used but need to be passed
                    empty_img, empty_obj,
                    intrinsic_1, dist_1,
                    empty_img, empty_obj,
                    local_intrinsic, local_dist,
                    single_frame_img, single_frame_obj,
                    dummy_qvec, dummy_tvec, // cam1 (not used)
                    dummy_qvec, dummy_tvec, // cam2 (not used)
                    single_target_pose,
                    single_timestamp,
                    single_cam_flags,
                    true, // fix cam1
                    true  // fix cam2
                );
            }
            
            // Save calibration results AFTER optimization
            {
                // Prepare intrinsics arrays for save (use current frame intrinsics)
                double save_intrinsic_0[4] = {frame_intrinsic_0[0], frame_intrinsic_0[1], frame_intrinsic_0[2], frame_intrinsic_0[3]};
                double save_dist_0[4] = {frame_dist_0[0], frame_dist_0[1], frame_dist_0[2], frame_dist_0[3]};
                double save_intrinsic_1[4] = {frame_intrinsic_1[0], frame_intrinsic_1[1], frame_intrinsic_1[2], frame_intrinsic_1[3]};
                double save_dist_1[4] = {frame_dist_1[0], frame_dist_1[1], frame_dist_1[2], frame_dist_1[3]};
                double save_intrinsic_2[4] = {frame_intrinsic_2[0], frame_intrinsic_2[1], frame_intrinsic_2[2], frame_intrinsic_2[3]};
                double save_dist_2[4] = {frame_dist_2[0], frame_dist_2[1], frame_dist_2[2], frame_dist_2[3]};
                
                // Update the camera being optimized with optimized local values
                if (cam_id == 0) {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_0);
                    std::copy(local_dist, local_dist + 4, save_dist_0);
                } else if (cam_id == 1) {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_1);
                    std::copy(local_dist, local_dist + 4, save_dist_1);
                } else {
                    std::copy(local_intrinsic, local_intrinsic + 4, save_intrinsic_2);
                    std::copy(local_dist, local_dist + 4, save_dist_2);
                }
                
                // Use optimized pose for this camera (single_timestamp has one entry, so one target pose)
                std::vector<std::array<double, 7>> frame_target_poses_after = {single_target_pose[0]};
                
                std::string filename_after = "calibration_result_timestamp_" + std::to_string(entry.timestamp_id) + "_cam" + std::to_string(cam_id) + "_after.json";
                SaveCalibrationResult(
                    filename_after,
                    save_intrinsic_0, save_dist_0,
                    save_intrinsic_1, save_dist_1,
                    save_intrinsic_2, save_dist_2,
                    dummy_qvec, dummy_tvec, // cam1 (dummy)
                    dummy_qvec, dummy_tvec, // cam2 (dummy)
                    frame_target_poses_after,
                    single_timestamp
                );
                std::cout << "    Saved calibration results AFTER optimization to " << filename_after << std::endl;
            }
            
            // Store optimized intrinsics and target pose
            if (cam_id == 0) {
                per_frame_intrinsics_0.push_back({local_intrinsic[0], local_intrinsic[1], local_intrinsic[2], local_intrinsic[3]});
                per_frame_dist_0.push_back({local_dist[0], local_dist[1], local_dist[2], local_dist[3]});
                // Update frame-specific intrinsics
                for (int i = 0; i < 4; ++i) {
                    frame_intrinsic_0[i] = local_intrinsic[i];
                    frame_dist_0[i] = local_dist[i];
                }
            } else if (cam_id == 1) {
                per_frame_intrinsics_1.push_back({local_intrinsic[0], local_intrinsic[1], local_intrinsic[2], local_intrinsic[3]});
                per_frame_dist_1.push_back({local_dist[0], local_dist[1], local_dist[2], local_dist[3]});
                // Update frame-specific intrinsics
                for (int i = 0; i < 4; ++i) {
                    frame_intrinsic_1[i] = local_intrinsic[i];
                    frame_dist_1[i] = local_dist[i];
                }
            } else {
                per_frame_intrinsics_2.push_back({local_intrinsic[0], local_intrinsic[1], local_intrinsic[2], local_intrinsic[3]});
                per_frame_dist_2.push_back({local_dist[0], local_dist[1], local_dist[2], local_dist[3]});
                // Update frame-specific intrinsics
                for (int i = 0; i < 4; ++i) {
                    frame_intrinsic_2[i] = local_intrinsic[i];
                    frame_dist_2[i] = local_dist[i];
                }
            }
            
            per_frame_target_poses_by_cam[frame_idx].push_back(single_target_pose[0]);
            
            std::cout << "    Camera " << cam_id << " optimized. Intrinsics: fx=" << local_intrinsic[0] 
                      << ", fy=" << local_intrinsic[1] << ", cx=" << local_intrinsic[2] 
                      << ", cy=" << local_intrinsic[3] << std::endl;
        }
        
        // Save calibration results for this frame after all cameras are processed
        {
            // Collect target poses for this frame (one per camera that was optimized)
            std::vector<std::array<double, 7>> frame_target_poses;
            const auto& target_poses_cam = per_frame_target_poses_by_cam[frame_idx];
            for (const auto& tp : target_poses_cam) {
                frame_target_poses.push_back(tp);
            }
            
            // Create timestamp entry for this frame
            std::vector<TimestampEntry> frame_timestamp = {entry};
            
            // Use dummy extrinsics (identity) since we're not optimizing them yet
            double dummy_qvec[4] = {1.0, 0.0, 0.0, 0.0};
            double dummy_tvec[3] = {0.0, 0.0, 0.0};
            
            // Save calibration result for this frame
            std::string filename = "calibration_result_initial_timestamp_" + std::to_string(entry.timestamp_id) + ".json";
            SaveCalibrationResult(
                filename,
                frame_intrinsic_0, frame_dist_0,
                frame_intrinsic_1, frame_dist_1,
                frame_intrinsic_2, frame_dist_2,
                dummy_qvec, dummy_tvec, // cam1 (dummy)
                dummy_qvec, dummy_tvec, // cam2 (dummy)
                frame_target_poses,
                frame_timestamp
            );
            std::cout << "  Saved calibration results for timestamp " << entry.timestamp_id << " to " << filename << std::endl;
        }
    }
    // std::cin.get();
    // Step 3a: Average intrinsics across frames for each camera
    std::cout << "\n========== Averaging intrinsics across frames ==========" << std::endl;
    
    if (!per_frame_intrinsics_0.empty()) {
        std::array<double, 4> avg_intrinsic = {0, 0, 0, 0};
        std::array<double, 4> avg_dist = {0, 0, 0, 0};
        for (const auto& intr : per_frame_intrinsics_0) {
            for (int j = 0; j < 4; ++j) avg_intrinsic[j] += intr[j];
        }
        for (const auto& dist : per_frame_dist_0) {
            for (int j = 0; j < 4; ++j) avg_dist[j] += dist[j];
        }
        double n = per_frame_intrinsics_0.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_0[j] = avg_intrinsic[j] / n;
            dist_0[j] = avg_dist[j] / n;
        }
        std::cout << "Camera 0: Averaged " << n << " frame estimates" << std::endl;
    }
    
    if (!per_frame_intrinsics_1.empty()) {
        std::array<double, 4> avg_intrinsic = {0, 0, 0, 0};
        std::array<double, 4> avg_dist = {0, 0, 0, 0};
        for (const auto& intr : per_frame_intrinsics_1) {
            for (int j = 0; j < 4; ++j) avg_intrinsic[j] += intr[j];
        }
        for (const auto& dist : per_frame_dist_1) {
            for (int j = 0; j < 4; ++j) avg_dist[j] += dist[j];
        }
        double n = per_frame_intrinsics_1.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_1[j] = avg_intrinsic[j] / n;
            dist_1[j] = avg_dist[j] / n;
        }
        std::cout << "Camera 1: Averaged " << n << " frame estimates" << std::endl;
    }
    
    if (!per_frame_intrinsics_2.empty()) {
        std::array<double, 4> avg_intrinsic = {0, 0, 0, 0};
        std::array<double, 4> avg_dist = {0, 0, 0, 0};
        for (const auto& intr : per_frame_intrinsics_2) {
            for (int j = 0; j < 4; ++j) avg_intrinsic[j] += intr[j];
        }
        for (const auto& dist : per_frame_dist_2) {
            for (int j = 0; j < 4; ++j) avg_dist[j] += dist[j];
        }
        double n = per_frame_intrinsics_2.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_2[j] = avg_intrinsic[j] / n;
            dist_2[j] = avg_dist[j] / n;
        }
        std::cout << "Camera 2: Averaged " << n << " frame estimates" << std::endl;
    }
    
    // Step 3a (continued): Compute and average inter-camera extrinsics
    std::cout << "\n========== Computing inter-camera extrinsics ==========" << std::endl;
    
    // Add transformation parameters for camera 1 and camera 2 (relative to camera 0)
    double rvec_cam_1[3];
    double tvec_cam_1[3];
    double qvec_cam_1[4];
    double rvec_cam_2[3];
    double tvec_cam_2[3];
    double qvec_cam_2[4];
    
    // Load inter-camera extrinsics from file if provided
    bool extrinsics_loaded = false;
    if (!extrinsics_file.empty()) {
        extrinsics_loaded = LoadExtrinsicsFromJson(extrinsics_file,
                                                    qvec_cam_1, tvec_cam_1,
                                                    qvec_cam_2, tvec_cam_2);
    }
    
    if (!extrinsics_loaded) {
        // Compute inter-camera extrinsics from per-frame target poses
        std::vector<Eigen::Matrix4d> cam1_to_cam0_list, cam2_to_cam0_list;
        // Per-frame storage for saving pre-optimization extrinsics (frame_index -> present?, T)
        std::vector<std::pair<bool, Eigen::Matrix4d>> cam1_per_frame(master_timestamps.size(), {false, Eigen::Matrix4d::Identity()});
        std::vector<std::pair<bool, Eigen::Matrix4d>> cam2_per_frame(master_timestamps.size(), {false, Eigen::Matrix4d::Identity()});
        std::vector<std::pair<bool, Eigen::Matrix4d>> cam2_to_cam1_per_frame(master_timestamps.size(), {false, Eigen::Matrix4d::Identity()});

        for (size_t frame_idx = 0; frame_idx < master_timestamps.size(); ++frame_idx) {
            const auto& entry = master_timestamps[frame_idx];
            const auto& target_poses_cam = per_frame_target_poses_by_cam[frame_idx];
            
            // Find target poses for each camera in this frame
            std::array<double,7> target_pose_cam0, target_pose_cam1, target_pose_cam2;
            bool has_cam0 = false, has_cam1 = false, has_cam2 = false;
            
            int cam_idx = 0;
            if (entry.cam0_idx != -1 && cam_idx < target_poses_cam.size()) {
                target_pose_cam0 = target_poses_cam[cam_idx++];
                has_cam0 = true;
            }
            if (entry.cam1_idx != -1 && cam_idx < target_poses_cam.size()) {
                target_pose_cam1 = target_poses_cam[cam_idx++];
                has_cam1 = true;
            }
            if (entry.cam2_idx != -1 && cam_idx < target_poses_cam.size()) {
                target_pose_cam2 = target_poses_cam[cam_idx++];
                has_cam2 = true;
            }
            
            // Compute cam1→cam0: T_cam1_in_cam0 = T_target_in_cam0 * T_target_in_cam1^-1
            if (has_cam0 && has_cam1) {
                Eigen::Matrix4d T_target_in_cam0 = arrayPoseToMatrix(target_pose_cam0);
                Eigen::Matrix4d T_target_in_cam1 = arrayPoseToMatrix(target_pose_cam1);
                Eigen::Matrix4d T_cam1_in_cam0 = T_target_in_cam0 * T_target_in_cam1.inverse();
                cam1_to_cam0_list.push_back(T_cam1_in_cam0);
                cam1_per_frame[frame_idx] = {true, T_cam1_in_cam0};
            }
            
            // Compute cam2→cam0: T_cam2_in_cam0 = T_target_in_cam0 * T_target_in_cam2^-1
            if (has_cam0 && has_cam2) {
                Eigen::Matrix4d T_target_in_cam0 = arrayPoseToMatrix(target_pose_cam0);
                Eigen::Matrix4d T_target_in_cam2 = arrayPoseToMatrix(target_pose_cam2);
                Eigen::Matrix4d T_cam2_in_cam0 = T_target_in_cam0 * T_target_in_cam2.inverse();
                cam2_to_cam0_list.push_back(T_cam2_in_cam0);
                cam2_per_frame[frame_idx] = {true, T_cam2_in_cam0};
            }
            
            // Compute cam2→cam1: T_cam2_in_cam1 = T_target_in_cam1 * T_target_in_cam2^-1 (for frames with cam1+cam2 but possibly no cam0)
            if (has_cam1 && has_cam2) {
                Eigen::Matrix4d T_target_in_cam1 = arrayPoseToMatrix(target_pose_cam1);
                Eigen::Matrix4d T_target_in_cam2 = arrayPoseToMatrix(target_pose_cam2);
                Eigen::Matrix4d T_cam2_in_cam1 = T_target_in_cam1 * T_target_in_cam2.inverse();
                cam2_to_cam1_per_frame[frame_idx] = {true, T_cam2_in_cam1};
            }
        }

        // Save pre-optimization per-frame extrinsics to JSON for consistency analysis.
        // When running one frame per command (e.g. rc3.sh), merge with existing file by timestamp_id
        // so all runs accumulate into one JSON instead of overwriting.
        {
            json pre_opt_json;
            pre_opt_json["description"] = "Pre-optimization inter-camera extrinsics per frame (from single-camera target poses, before global optimization)";
            pre_opt_json["frames"] = json::array();
            for (size_t i = 0; i < master_timestamps.size(); ++i) {
                json frame_entry;
                frame_entry["frame_index"] = static_cast<int>(i);
                frame_entry["timestamp_id"] = master_timestamps[i].timestamp_id;
                if (cam1_per_frame[i].first) {
                    const Eigen::Matrix4d& T = cam1_per_frame[i].second;
                    Eigen::Quaterniond q(T.block<3,3>(0,0));
                    frame_entry["camera1_to_camera0"] = {
                        {"quaternion", json::array({q.w(), q.x(), q.y(), q.z()})},
                        {"translation", json::array({T(0,3), T(1,3), T(2,3)})}
                    };
                } else {
                    frame_entry["camera1_to_camera0"] = nullptr;
                }
                if (cam2_per_frame[i].first) {
                    const Eigen::Matrix4d& T = cam2_per_frame[i].second;
                    Eigen::Quaterniond q(T.block<3,3>(0,0));
                    frame_entry["camera2_to_camera0"] = {
                        {"quaternion", json::array({q.w(), q.x(), q.y(), q.z()})},
                        {"translation", json::array({T(0,3), T(1,3), T(2,3)})}
                    };
                } else {
                    frame_entry["camera2_to_camera0"] = nullptr;
                }
                if (cam2_to_cam1_per_frame[i].first) {
                    const Eigen::Matrix4d& T = cam2_to_cam1_per_frame[i].second;
                    Eigen::Quaterniond q(T.block<3,3>(0,0));
                    frame_entry["camera2_to_camera1"] = {
                        {"quaternion", json::array({q.w(), q.x(), q.y(), q.z()})},
                        {"translation", json::array({T(0,3), T(1,3), T(2,3)})}
                    };
                } else {
                    frame_entry["camera2_to_camera1"] = nullptr;
                }
                pre_opt_json["frames"].push_back(frame_entry);
            }

            std::string pre_opt_path = "pre_optimization_extrinsics_per_frame.json";
            json merged = pre_opt_json;
            std::ifstream ifs(pre_opt_path);
            if (ifs && ifs.good()) {
                try {
                    json existing;
                    ifs >> existing;
                    if (existing.contains("frames") && existing["frames"].is_array()) {
                        json::array_t& existing_frames = existing["frames"].get_ref<json::array_t&>();
                        for (const json& new_frame : pre_opt_json["frames"]) {
                            int ts_id = new_frame["timestamp_id"].get<int>();
                            bool found = false;
                            for (size_t k = 0; k < existing_frames.size(); ++k) {
                                if (existing_frames[k].contains("timestamp_id") &&
                                    existing_frames[k]["timestamp_id"].get<int>() == ts_id) {
                                    existing_frames[k] = new_frame;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                existing_frames.push_back(new_frame);
                            }
                        }
                        merged["frames"] = existing_frames;
                        // Sort by timestamp_id and renumber frame_index
                        json::array_t& frames = merged["frames"].get_ref<json::array_t&>();
                        std::sort(frames.begin(), frames.end(),
                            [](const json& a, const json& b) {
                                return a["timestamp_id"].get<int>() < b["timestamp_id"].get<int>();
                            });
                        for (size_t k = 0; k < frames.size(); ++k) {
                            frames[k]["frame_index"] = static_cast<int>(k);
                        }
                    }
                } catch (const json::exception& e) {
                    std::cerr << "Warning: Could not merge existing " << pre_opt_path << " (" << e.what() << "), writing current run only." << std::endl;
                }
            }

            std::ofstream ofs(pre_opt_path);
            if (ofs) {
                ofs << std::setw(4) << merged << std::endl;
                std::cout << "Saved pre-optimization per-frame extrinsics to " << pre_opt_path
                          << " (" << merged["frames"].size() << " frames)" << std::endl;
            } else {
                std::cerr << "Warning: Could not write " << pre_opt_path << std::endl;
            }
        }
        
        // Average inter-camera extrinsics
        if (!cam1_to_cam0_list.empty()) {
            Eigen::Matrix4d cam1_in_cam0_avg = averagePoses(cam1_to_cam0_list);
            Eigen::Quaterniond q1(cam1_in_cam0_avg.block<3,3>(0,0));
            qvec_cam_1[0] = q1.w(); qvec_cam_1[1] = q1.x(); qvec_cam_1[2] = q1.y(); qvec_cam_1[3] = q1.z();
            tvec_cam_1[0] = cam1_in_cam0_avg(0,3);
            tvec_cam_1[1] = cam1_in_cam0_avg(1,3);
            tvec_cam_1[2] = cam1_in_cam0_avg(2,3);
            std::cout << "Camera 1→Camera 0: Averaged " << cam1_to_cam0_list.size() << " estimates" << std::endl;
        } else {
            // Fallback
            qvec_cam_1[0] = 1.0; qvec_cam_1[1] = 0.0; qvec_cam_1[2] = 0.0; qvec_cam_1[3] = 0.0;
            tvec_cam_1[0] = 0.1; tvec_cam_1[1] = 0.1; tvec_cam_1[2] = 0.0;
        }
        
        if (!cam2_to_cam0_list.empty()) {
            Eigen::Matrix4d cam2_in_cam0_avg = averagePoses(cam2_to_cam0_list);
            Eigen::Quaterniond q2(cam2_in_cam0_avg.block<3,3>(0,0));
            qvec_cam_2[0] = q2.w(); qvec_cam_2[1] = q2.x(); qvec_cam_2[2] = q2.y(); qvec_cam_2[3] = q2.z();
            tvec_cam_2[0] = cam2_in_cam0_avg(0,3);
            tvec_cam_2[1] = cam2_in_cam0_avg(1,3);
            tvec_cam_2[2] = cam2_in_cam0_avg(2,3);
            std::cout << "Camera 2→Camera 0: Averaged " << cam2_to_cam0_list.size() << " estimates" << std::endl;
        } else {
            // Fallback
            qvec_cam_2[0] = 1.0; qvec_cam_2[1] = 0.0; qvec_cam_2[2] = 0.0; qvec_cam_2[3] = 0.0;
            tvec_cam_2[0] = 0.2; tvec_cam_2[1] = 0.0; tvec_cam_2[2] = 0.0;
        }
    }
    
    ceres::QuaternionToAngleAxis(qvec_cam_1, rvec_cam_1);
    ceres::QuaternionToAngleAxis(qvec_cam_2, rvec_cam_2);
    
    // Step 3b: For each frame, average target poses in global frame (cam0 frame)
    std::cout << "\n========== Averaging target poses in global frame ==========" << std::endl;
    
    std::vector<std::array<double,7>> target_poses;
    target_poses.reserve(master_timestamps.size());
    
    Eigen::Matrix4d T_cam1_in_cam0 = quatTransToMatrix(qvec_cam_1, tvec_cam_1);
    Eigen::Matrix4d T_cam2_in_cam0 = quatTransToMatrix(qvec_cam_2, tvec_cam_2);
    
    for (size_t frame_idx = 0; frame_idx < master_timestamps.size(); ++frame_idx) {
        const auto& entry = master_timestamps[frame_idx];
        const auto& target_poses_cam = per_frame_target_poses_by_cam[frame_idx];
        
        std::vector<Eigen::Matrix4d> target_pose_estimates;
        
        // Find target poses for each camera in this frame
        int cam_idx = 0;
        if (entry.cam0_idx != -1 && cam_idx < target_poses_cam.size()) {
            // Target pose already in cam0 frame
            target_pose_estimates.push_back(arrayPoseToMatrix(target_poses_cam[cam_idx++]));
        }
        if (entry.cam1_idx != -1 && cam_idx < target_poses_cam.size()) {
            // Transform target pose from cam1 frame to cam0 frame
            Eigen::Matrix4d T_target_in_cam1 = arrayPoseToMatrix(target_poses_cam[cam_idx++]);
            Eigen::Matrix4d T_target_in_cam0 = T_cam1_in_cam0 * T_target_in_cam1;
            target_pose_estimates.push_back(T_target_in_cam0);
        }
        if (entry.cam2_idx != -1 && cam_idx < target_poses_cam.size()) {
            // Transform target pose from cam2 frame to cam0 frame
            Eigen::Matrix4d T_target_in_cam2 = arrayPoseToMatrix(target_poses_cam[cam_idx++]);
            Eigen::Matrix4d T_target_in_cam0 = T_cam2_in_cam0 * T_target_in_cam2;
            target_pose_estimates.push_back(T_target_in_cam0);
        }
        
        if (!target_pose_estimates.empty()) {
            Eigen::Matrix4d T_avg = averagePoses(target_pose_estimates);
            target_poses.push_back(matrixToArrayPose(T_avg));
        } else {
            // Fallback: identity
            target_poses.push_back({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        }
    }
    
    std::cout << "Initialized " << target_poses.size() << " target poses in global frame (cam0)." << std::endl;
    
    SaveCalibrationResult("calibration_result_initial.json",
        intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses, master_timestamps
    );
    
    std::cout << "\n========== NEW INITIALIZATION COMPLETE ==========\n" << std::endl;
    // ========== END NEW INITIALIZATION SCHEMA ==========

    // ========== OLD INITIALIZATION CODE (COMMENTED OUT - REPLACED BY NEW INITIALIZATION) ==========
    // --- START: single-target-pose initialization (drop-in) --------------------
    /*
    auto quatTransToMatrix = [](const double quat_arr[4], const double t_arr[3]) {
        Eigen::Quaterniond q(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3]); // [w,x,y,z]
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T.block<3,1>(0,3) = Eigen::Vector3d(t_arr[0], t_arr[1], t_arr[2]);
        return T;
    };


    // --- Helpers ---
    auto arrayPoseToMatrix = [](const std::array<double,7>& a) {
        Eigen::Quaterniond q(a[0], a[1], a[2], a[3]); // [w,x,y,z]
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T.block<3,1>(0,3) = Eigen::Vector3d(a[4], a[5], a[6]);
        return T;
    };

    auto matrixToArrayPose = [](const Eigen::Matrix4d& T) {
        Eigen::Quaterniond q(T.block<3,3>(0,0));
        std::array<double,7> a;
        a[0]=q.w(); a[1]=q.x(); a[2]=q.y(); a[3]=q.z();
        a[4]=T(0,3); a[5]=T(1,3); a[6]=T(2,3);
        return a;
    };

    // Log/exp helpers for averaging rotations
    auto logSO3 = [](const Eigen::Matrix3d& R) {
        Eigen::AngleAxisd aa(R);
        return aa.angle() * aa.axis();
    };
    auto expSO3 = [](const Eigen::Vector3d& r) -> Eigen::Matrix3d {
        double theta = r.norm();
        if (theta < 1e-12) return Eigen::Matrix3d::Identity();
        Eigen::AngleAxisd aa(theta, r.normalized());
        return aa.toRotationMatrix();
    };

    // Average SE(3) transforms
    auto averagePoses = [&](const std::vector<Eigen::Matrix4d>& Ts) {
        Eigen::Vector3d t_avg = Eigen::Vector3d::Zero();
        Eigen::Vector3d r_accum = Eigen::Vector3d::Zero();
        for (const auto& T : Ts) {
            t_avg += T.block<3,1>(0,3);
            r_accum += logSO3(T.block<3,3>(0,0));
        }
        t_avg /= Ts.size();
        r_accum /= Ts.size();
        Eigen::Matrix4d T_avg = Eigen::Matrix4d::Identity();
        T_avg.block<3,3>(0,0) = expSO3(r_accum);
        T_avg.block<3,1>(0,3) = t_avg;
        return T_avg;
    };

    // --- Step 1: Camera->Target extrinsics are already computed as extrinsics_0,1,2 ---

    // --- Step 2: Estimate inter-camera transforms ---
    std::vector<Eigen::Matrix4d> cam0_to_cam1_list, cam1_to_cam2_list;

    // For frames where cam0 and cam1 both saw the board
    for (size_t i = 0; i < master_timestamps.size(); ++i) {
        const auto& entry = master_timestamps[i];
        if (entry.cam0_idx != -1 && entry.cam1_idx != -1) {
            Eigen::Matrix4d T_t_in_c0 = arrayPoseToMatrix(extrinsics_0[entry.cam0_idx]); // target in cam0
            Eigen::Matrix4d T_t_in_c1 = arrayPoseToMatrix(extrinsics_1[entry.cam1_idx]); // target in cam1
            Eigen::Matrix4d T_c1_in_c0 = T_t_in_c0 * T_t_in_c1.inverse();
            cam0_to_cam1_list.push_back(T_c1_in_c0);
        }
    }

    // Average these
    Eigen::Matrix4d cam1_in_cam0 = averagePoses(cam0_to_cam1_list);

    // For frames where cam1 and cam2 both saw the board
    for (size_t i = 0; i < master_timestamps.size(); ++i) {
        const auto& entry = master_timestamps[i];
        if (entry.cam1_idx != -1 && entry.cam2_idx != -1) {
            Eigen::Matrix4d T_t_in_c1 = arrayPoseToMatrix(extrinsics_1[entry.cam1_idx]);
            Eigen::Matrix4d T_t_in_c2 = arrayPoseToMatrix(extrinsics_2[entry.cam2_idx]);
            Eigen::Matrix4d T_c2_in_c1 = T_t_in_c1 * T_t_in_c2.inverse();
            cam1_to_cam2_list.push_back(T_c2_in_c1);
        }
    }

    Eigen::Matrix4d cam2_in_cam1 = averagePoses(cam1_to_cam2_list);
    Eigen::Matrix4d cam2_in_cam0 = cam1_in_cam0 * cam2_in_cam1;
    
    // Load or compute inter-camera extrinsics
    bool extrinsics_loaded = false;
    if (!extrinsics_file.empty()) {
        extrinsics_loaded = LoadExtrinsicsFromJson(extrinsics_file,
                                                    qvec_cam_1, tvec_cam_1,
                                                    qvec_cam_2, tvec_cam_2);
        if (!extrinsics_loaded) {
            std::cout << "Warning: Failed to load extrinsics from file, using computed values." << std::endl;
        }
    }
    
    // If not loaded from file, use the computed values from averaging
    if (!extrinsics_loaded) {
        // Check if we have valid computed transforms (non-empty lists)
        bool has_valid_computation = !cam0_to_cam1_list.empty() && !cam1_to_cam2_list.empty();
        
        if (has_valid_computation) {
            // Convert computed transforms to quaternion and translation
            Eigen::Quaterniond q1(cam1_in_cam0.block<3,3>(0,0));
            qvec_cam_1[0] = q1.w(); qvec_cam_1[1] = q1.x(); qvec_cam_1[2] = q1.y(); qvec_cam_1[3] = q1.z();
            tvec_cam_1[0] = cam1_in_cam0(0,3); tvec_cam_1[1] = cam1_in_cam0(1,3); tvec_cam_1[2] = cam1_in_cam0(2,3);
            
            Eigen::Quaterniond q2(cam2_in_cam0.block<3,3>(0,0));
            qvec_cam_2[0] = q2.w(); qvec_cam_2[1] = q2.x(); qvec_cam_2[2] = q2.y(); qvec_cam_2[3] = q2.z();
            tvec_cam_2[0] = cam2_in_cam0(0,3); tvec_cam_2[1] = cam2_in_cam0(1,3); tvec_cam_2[2] = cam2_in_cam0(2,3);
            
            std::cout << "Using computed inter-camera extrinsics from target poses." << std::endl;
        } else {
            // Fallback to default hardcoded values if computation failed
            std::cout << "Warning: Could not compute extrinsics from target poses (no overlapping frames), using defaults." << std::endl;
            double rvec_cam_1_default[3] = {0, 3.14/3, 0.0};
            double tvec_cam_1_default[3] = {.1, .1, 0};
            ceres::AngleAxisToQuaternion(rvec_cam_1_default, qvec_cam_1);
            tvec_cam_1[0] = tvec_cam_1_default[0];
            tvec_cam_1[1] = tvec_cam_1_default[1];
            tvec_cam_1[2] = tvec_cam_1_default[2];
            
            double rvec_cam_2_default[3] = {0, 2*3.14/3, 0};
            double tvec_cam_2_default[3] = {.2, -0.0, -0.0};
            ceres::AngleAxisToQuaternion(rvec_cam_2_default, qvec_cam_2);
            tvec_cam_2[0] = tvec_cam_2_default[0];
            tvec_cam_2[1] = tvec_cam_2_default[1];
            tvec_cam_2[2] = tvec_cam_2_default[2];
        }
    }
    
    // Convert quaternions to angle-axis for display (optional, for compatibility)
    ceres::QuaternionToAngleAxis(qvec_cam_1, rvec_cam_1);
    ceres::QuaternionToAngleAxis(qvec_cam_2, rvec_cam_2);

    // --- Step 3: Build target poses in cam0 frame ---
    std::vector<std::array<double,7>> target_poses;
    target_poses.reserve(master_timestamps.size());

    for (const auto& entry : master_timestamps) {
        std::vector<Eigen::Matrix4d> estimates;

        if (entry.cam0_idx != -1) {
            estimates.push_back(arrayPoseToMatrix(extrinsics_0[entry.cam0_idx]));
        }
        if (entry.cam1_idx != -1) {
            Eigen::Matrix4d T_t_in_c1 = arrayPoseToMatrix(extrinsics_1[entry.cam1_idx]);
            estimates.push_back(cam1_in_cam0 * T_t_in_c1);
        }
        if (entry.cam2_idx != -1) {
            Eigen::Matrix4d T_t_in_c2 = arrayPoseToMatrix(extrinsics_2[entry.cam2_idx]);
            estimates.push_back(cam2_in_cam0 * T_t_in_c2);
        }

        if (!estimates.empty()) {
            Eigen::Matrix4d T_avg = averagePoses(estimates);
            target_poses.push_back(matrixToArrayPose(T_avg));
        } else {
            // fallback: identity
            target_poses.push_back({1.0,0.0,0.0,0.0,0.0,0.0,0.0});
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

    // //Load known simulated target poses from JSON
    // std::vector<std::array<double,7>> target_poses;
    // std::vector<double> timestamps;
    // if (LoadTargetPosesFromJson(target_poses_file, target_poses, &timestamps)) {
    //     std::cout << "Loaded " << target_poses.size() << " target poses from JSON\n";
    //     // If you need them as Eigen matrices
    //     std::vector<Eigen::Matrix4d> target_mats;
    //     target_mats.reserve(target_poses.size());
    //     for (const auto& a : target_poses) {
    //         target_mats.push_back(quatTransToMatrixLoaded(a));
    //     }
    //     // Now use target_poses (vector of arrays) and target_mats directly in subsequent code,
    //     // bypassing the earlier "estimate inter-camera and average" logic.
    // } else {
    //     std::cerr << "Failed to load simulated poses; falling back to estimation path\n";
    //     // keep your original estimation code here
    // }

    //output target poses for verification
    for (size_t i = 0; i < target_poses.size(); ++i) {
        const auto& tp = target_poses[i];
        std::cout << "Target Pose " << i << ": ["
                  << tp[0] << ", " << tp[1] << ", " << tp[2] << ", " << tp[3] << "] , ["
                  << tp[4] << ", " << tp[5] << ", " << tp[6] << "]\n";
    }
    // also output inter-camera transforms
    std::cout << "Inter-camera Transform Camera1 to Camera0: Quaternion ["
              << qvec_cam_1[0] << ", " << qvec_cam_1[1] << ", " << qvec_cam_1[2] << ", " << qvec_cam_1[3]
              << "], Translation ["
              << tvec_cam_1[0] << ", " << tvec_cam_1[1] << ", " << tvec_cam_1[2] << "]\n";
    std::cout << "Inter-camera Transform Camera2 to Camera0: Quaternion ["
              << qvec_cam_2[0] << ", " << qvec_cam_2[1] << ", " << qvec_cam_2[2] << ", " << qvec_cam_2[3]
              << "], Translation ["
              << tvec_cam_2[0] << ", " << tvec_cam_2[1] << ", " << tvec_cam_2[2] << "]\n";

    // --- Validate and fix target poses that don't produce valid projections ---
    
    // Helper: Create "right in front" pose in a specific camera's frame, then transform to cam0
    auto createRightInFrontPoseInCam0 = [&](int cam_id) -> std::array<double,7> {
        // Create pose in the specified camera's frame: identity rotation, 0.5m in front along Z
        Eigen::Quaterniond q_cam(1.0, 0.0, 0.0, 0.0); // identity rotation
        Eigen::Vector3d t_cam(0.0, 0.0, 0.5); // 0.5m in front
        
        Eigen::Matrix4d T_target_in_cam = Eigen::Matrix4d::Identity();
        T_target_in_cam.block<3,3>(0,0) = q_cam.toRotationMatrix();
        T_target_in_cam.block<3,1>(0,3) = t_cam;
        
        // Transform to cam0 frame
        Eigen::Matrix4d T_target_in_cam0;
        if (cam_id == 0) {
            // Already in cam0 frame
            T_target_in_cam0 = T_target_in_cam;
        } else if (cam_id == 1) {
            // Transform: target_in_cam0 = cam1_in_cam0 * target_in_cam1
            Eigen::Matrix4d T_cam1_in_cam0 = quatTransToMatrix(qvec_cam_1, tvec_cam_1);
            T_target_in_cam0 = T_cam1_in_cam0 * T_target_in_cam;
        } else if (cam_id == 2) {
            // Transform: target_in_cam0 = cam2_in_cam0 * target_in_cam2
            Eigen::Matrix4d T_cam2_in_cam0 = quatTransToMatrix(qvec_cam_2, tvec_cam_2);
            T_target_in_cam0 = T_cam2_in_cam0 * T_target_in_cam;
        } else {
            // Fallback: identity
            T_target_in_cam0 = Eigen::Matrix4d::Identity();
            T_target_in_cam0(2,3) = 0.5; // 0.5m in front
        }
        
        return matrixToArrayPose(T_target_in_cam0);
    };
    
    // Helper: Validate target pose produces valid projections
    auto validateTargetPose = [&](size_t frame_idx, const std::array<double,7>& target_pose) -> bool {
        if (frame_idx >= master_timestamps.size()) return false;
        
        const auto& entry = master_timestamps[frame_idx];
        
        // Convert target pose to matrix
        Eigen::Matrix4d T_target_in_cam0 = arrayPoseToMatrix(target_pose);
        Eigen::Matrix3d R_target = T_target_in_cam0.block<3,3>(0,0);
        Eigen::Vector3d t_target = T_target_in_cam0.block<3,1>(0,3);
        
        // Check each camera that observed this frame
        bool has_valid_projection = false;
        
        // CAM0
        if (entry.cam0_idx != -1 && !has_valid_projection) {
            int cam0_i = entry.cam0_idx;
            if (cam0_i < obj_pts_list_0.size() && cam0_i < img_pts_list_0.size()) {
                const auto& obj_pts = obj_pts_list_0[cam0_i];
                
                // Transform object points to cam0 frame: target→cam0
                Eigen::MatrixXd points_cam0(obj_pts.size(), 3);
                for (size_t j = 0; j < obj_pts.size(); ++j) {
                    Eigen::Vector3d pt_cam0 = R_target * obj_pts[j] + t_target;
                    points_cam0.row(j) = pt_cam0.transpose();
                }
                
                // Project using Kannala-Brandt
                Eigen::Vector4d K(intrinsic_0[0], intrinsic_0[1], intrinsic_0[2], intrinsic_0[3]);
                Eigen::Vector4d D(dist_0[0], dist_0[1], dist_0[2], dist_0[3]);
                Eigen::MatrixXd projected = kannala_brandt_project(points_cam0, K, D);
                
                // Check if at least one point is valid
                double img_width = 2.0 * intrinsic_0[2];
                double img_height = 2.0 * intrinsic_0[3];
                
                for (int j = 0; j < projected.rows(); ++j) {
                    double Z = points_cam0(j, 2);
                    double u = projected(j, 0);
                    double v = projected(j, 1);
                    
                    if (Z > 1e-6 && u >= 0 && u < img_width && v >= 0 && v < img_height) {
                        has_valid_projection = true;
                        break;
                    }
                }
            }
        }
        
        // CAM1
        if (entry.cam1_idx != -1 && !has_valid_projection) {
            int cam1_i = entry.cam1_idx;
            if (cam1_i < obj_pts_list_1.size() && cam1_i < img_pts_list_1.size()) {
                const auto& obj_pts = obj_pts_list_1[cam1_i];
                
                // Transform: target→cam0→cam1
                Eigen::Matrix4d T_cam1_in_cam0 = quatTransToMatrix(qvec_cam_1, tvec_cam_1);
                Eigen::Matrix4d T_cam0_in_cam1 = T_cam1_in_cam0.inverse();
                
                Eigen::MatrixXd points_cam1(obj_pts.size(), 3);
                for (size_t j = 0; j < obj_pts.size(); ++j) {
                    // First transform to cam0
                    Eigen::Vector3d pt_cam0 = R_target * obj_pts[j] + t_target;
                    // Then transform to cam1
                    Eigen::Vector4d pt_cam0_homog(pt_cam0.x(), pt_cam0.y(), pt_cam0.z(), 1.0);
                    Eigen::Vector4d pt_cam1_homog = T_cam0_in_cam1 * pt_cam0_homog;
                    points_cam1.row(j) = pt_cam1_homog.head<3>().transpose();
                }
                
                // Project using Kannala-Brandt
                Eigen::Vector4d K(intrinsic_1[0], intrinsic_1[1], intrinsic_1[2], intrinsic_1[3]);
                Eigen::Vector4d D(dist_1[0], dist_1[1], dist_1[2], dist_1[3]);
                Eigen::MatrixXd projected = kannala_brandt_project(points_cam1, K, D);
                
                // Check if at least one point is valid
                double img_width = 2.0 * intrinsic_1[2];
                double img_height = 2.0 * intrinsic_1[3];
                
                for (int j = 0; j < projected.rows(); ++j) {
                    double Z = points_cam1(j, 2);
                    double u = projected(j, 0);
                    double v = projected(j, 1);
                    
                    if (Z > 1e-6 && u >= 0 && u < img_width && v >= 0 && v < img_height) {
                        has_valid_projection = true;
                        break;
                    }
                }
            }
        }
        
        // CAM2
        if (entry.cam2_idx != -1 && !has_valid_projection) {
            int cam2_i = entry.cam2_idx;
            if (cam2_i < obj_pts_list_2.size() && cam2_i < img_pts_list_2.size()) {
                const auto& obj_pts = obj_pts_list_2[cam2_i];
                
                // Transform: target→cam0→cam2
                Eigen::Matrix4d T_cam2_in_cam0 = quatTransToMatrix(qvec_cam_2, tvec_cam_2);
                Eigen::Matrix4d T_cam0_in_cam2 = T_cam2_in_cam0.inverse();
                
                Eigen::MatrixXd points_cam2(obj_pts.size(), 3);
                for (size_t j = 0; j < obj_pts.size(); ++j) {
                    // First transform to cam0
                    Eigen::Vector3d pt_cam0 = R_target * obj_pts[j] + t_target;
                    // Then transform to cam2
                    Eigen::Vector4d pt_cam0_homog(pt_cam0.x(), pt_cam0.y(), pt_cam0.z(), 1.0);
                    Eigen::Vector4d pt_cam2_homog = T_cam0_in_cam2 * pt_cam0_homog;
                    points_cam2.row(j) = pt_cam2_homog.head<3>().transpose();
                }
                
                // Project using Kannala-Brandt
                Eigen::Vector4d K(intrinsic_2[0], intrinsic_2[1], intrinsic_2[2], intrinsic_2[3]);
                Eigen::Vector4d D(dist_2[0], dist_2[1], dist_2[2], dist_2[3]);
                Eigen::MatrixXd projected = kannala_brandt_project(points_cam2, K, D);
                
                // Check if at least one point is valid
                double img_width = 2.0 * intrinsic_2[2];
                double img_height = 2.0 * intrinsic_2[3];
                
                for (int j = 0; j < projected.rows(); ++j) {
                    double Z = points_cam2(j, 2);
                    double u = projected(j, 0);
                    double v = projected(j, 1);
                    
                    if (Z > 1e-6 && u >= 0 && u < img_width && v >= 0 && v < img_height) {
                        has_valid_projection = true;
                        break;
                    }
                }
            }
        }
        
        return has_valid_projection;
    };
    
    // Validate and fix target poses
    int fixed_count = 0;
    for (size_t i = 0; i < target_poses.size() && i < master_timestamps.size(); ++i) {
        const auto& entry = master_timestamps[i];
        
        if (!validateTargetPose(i, target_poses[i])) {
            // Find which camera has observations
            int cam_with_obs = -1;
            if (entry.cam0_idx != -1) {
                cam_with_obs = 0;
            } else if (entry.cam1_idx != -1) {
                cam_with_obs = 1;
            } else if (entry.cam2_idx != -1) {
                cam_with_obs = 2;
            }
            
            if (cam_with_obs != -1) {
                std::cout << "Warning: Frame " << i << " (timestamp " << entry.timestamp_id 
                          << ") target pose invalid, initializing to 'right in front' of CAM" 
                          << cam_with_obs << std::endl;
                target_poses[i] = createRightInFrontPoseInCam0(cam_with_obs);
                fixed_count++;
            } else {
                // No camera observed this frame - use identity as fallback
                std::cout << "Warning: Frame " << i << " has no camera observations, using identity pose." << std::endl;
                target_poses[i] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            }
        }
    }
    
    if (fixed_count > 0) {
        std::cout << "Fixed " << fixed_count << " invalid target pose(s) by initializing to 'right in front' of observing camera." << std::endl;
    }





    // Debug print (optional)
    std::cout << "Initialized " << target_poses.size() << " target_poses (in cam0 frame)." << std::endl;


    SaveCalibrationResult("calibration_result_initial.json",
        intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses, master_timestamps
    );

    // --- END: single-target-pose initialization --------------------------------
    */
    // ========== END OLD INITIALIZATION CODE ==========








    // --- Stage 1: Per-frame target pose refinement ---
    // Optimize each target pose independently using its available camera(s)
    std::cout << "Stage 1: Refining per-frame target poses, intrinsics, and extrinsics individually..." << std::endl;

    OptimizationFlags per_frame_flags;
    // Load per-frame flags from file if provided, otherwise use defaults
    bool per_frame_flags_loaded = false;
    if (!per_frame_flags_file.empty()) {
        per_frame_flags_loaded = LoadOptimizationFlagsFromJson(per_frame_flags_file, per_frame_flags);
        if (!per_frame_flags_loaded) {
            std::cout << "Warning: Failed to load per-frame flags from file, using defaults." << std::endl;
        }
    }
    
    if (!per_frame_flags_loaded) {
        // Default per-frame flags: optimize intrinsics and extrinsics per frame
        per_frame_flags.optimize_intrinsics = true;
        per_frame_flags.optimize_distortion = true;
        per_frame_flags.optimize_inter_camera = true;  // Will be set per frame based on camera visibility
        per_frame_flags.optimize_target_poses = true;
        std::cout << "Using default per-frame optimization flags:" << std::endl;
        std::cout << "  optimize_intrinsics: " << per_frame_flags.optimize_intrinsics << std::endl;
        std::cout << "  optimize_distortion: " << per_frame_flags.optimize_distortion << std::endl;
        std::cout << "  optimize_inter_camera: " << per_frame_flags.optimize_inter_camera << " (set per frame)" << std::endl;
        std::cout << "  optimize_target_poses: " << per_frame_flags.optimize_target_poses << std::endl;
    }

    // Storage for per-frame optimized intrinsics and extrinsics
    std::vector<std::array<double, 4>> frame_intrinsics_0, frame_intrinsics_1, frame_intrinsics_2;
    std::vector<std::array<double, 4>> frame_dist_0, frame_dist_1, frame_dist_2;
    std::vector<Eigen::Matrix4d> cam1_to_cam0_list, cam2_to_cam0_list;

    for (size_t i = 0; i < master_timestamps.size(); ++i) {
        // Build per-frame subsets
        std::vector<std::vector<Eigen::Vector2d>> img_pts_list_0_frame, img_pts_list_1_frame, img_pts_list_2_frame;
        std::vector<std::vector<Eigen::Vector3d>> obj_pts_list_0_frame, obj_pts_list_1_frame, obj_pts_list_2_frame;
        std::vector<TimestampEntry> single_timestamp = { master_timestamps[i] };
        //overwrite valid idexes to be 0 for single frame optimization
        if (single_timestamp[0].cam0_idx != -1) single_timestamp[0].cam0_idx = 0;
        if (single_timestamp[0].cam1_idx != -1) single_timestamp[0].cam1_idx = 0;
        if (single_timestamp[0].cam2_idx != -1) single_timestamp[0].cam2_idx = 0;
        //output single timestamp for verification
        std::cout << "Single_timestamp = { "
                  << "timestamp: " << single_timestamp[0].timestamp_id << ", "
                  << "cam0_idx: " << single_timestamp[0].cam0_idx << ", "
                  << "cam1_idx: " << single_timestamp[0].cam1_idx << ", "
                  << "cam2_idx: " << single_timestamp[0].cam2_idx << " }" << std::endl;

        // Fill per-camera data if available
        if (master_timestamps[i].cam0_idx != -1) {
            img_pts_list_0_frame.push_back(img_pts_list_0[master_timestamps[i].cam0_idx]);
            obj_pts_list_0_frame.push_back(obj_pts_list_0[master_timestamps[i].cam0_idx]);
        }
        if (master_timestamps[i].cam1_idx != -1) {
            img_pts_list_1_frame.push_back(img_pts_list_1[master_timestamps[i].cam1_idx]);
            obj_pts_list_1_frame.push_back(obj_pts_list_1[master_timestamps[i].cam1_idx]);
        }
        if (master_timestamps[i].cam2_idx != -1) {
            img_pts_list_2_frame.push_back(img_pts_list_2[master_timestamps[i].cam2_idx]);
            obj_pts_list_2_frame.push_back(obj_pts_list_2[master_timestamps[i].cam2_idx]);
        }
        //frame_target_poses
        std::vector<std::array<double,7>> frame_target_poses = { target_poses[i] };

        //output all image points and object points for verification
        // std::cout << "Frame " << i << " data:" << std::endl;
        // if (!img_pts_list_0_frame.empty()) {
        //     std::cout << "  Camera 0: " << img_pts_list_0_frame[0].size() << " points." << std::endl;
        //     for (const auto& pt : img_pts_list_0_frame[0]) {
        //         std::cout << "    Img Pt: [" << pt.x() << ", " << pt.y() << "]" << std::endl;
        //     }
        //     for (const auto& pt : obj_pts_list_0_frame[0]) {
        //         std::cout << "    Obj Pt: [" << pt.x() << ", " << pt.y() << ", " << pt.z() << "]" << std::endl;
        //     }
        // }
        // if (!img_pts_list_1_frame.empty()) {
        //     std::cout << "  Camera 1: " << img_pts_list_1_frame[0].size() << " points." << std::endl;
        //     for (const auto& pt : img_pts_list_1_frame[0]) {
        //         std::cout << "    Img Pt: [" << pt.x() << ", " << pt.y() << "]" << std::endl;
        //     }
        //     for (const auto& pt : obj_pts_list_1_frame[0]) {
        //         std::cout << "    Obj Pt: [" << pt.x() << ", " << pt.y() << ", " << pt.z() << "]" << std::endl;
        //     }
        // }
        // if (!img_pts_list_2_frame.empty()) {
        //     std::cout << "  Camera 2: " << img_pts_list_2_frame[0].size() << " points." << std::endl;
        //     for (const auto& pt : img_pts_list_2_frame[0]) {
        //         std::cout << "    Img Pt: [" << pt.x() << ", " << pt.y() << "]" << std::endl;
        //     }
        //     for (const auto& pt : obj_pts_list_2_frame[0]) {
        //         std::cout << "    Obj Pt: [" << pt.x() << ", " << pt.y() << ", " << pt.z() << "]" << std::endl;
        //     }
        // }

        // Determine which cameras are present and set extrinsics optimization accordingly
        bool has_cam0 = master_timestamps[i].cam0_idx != -1;
        bool has_cam1 = master_timestamps[i].cam1_idx != -1;
        bool has_cam2 = master_timestamps[i].cam2_idx != -1;
        
        int num_cameras = (has_cam0 ? 1 : 0) + (has_cam1 ? 1 : 0) + (has_cam2 ? 1 : 0);
        
        // Set extrinsics optimization based on camera visibility
        OptimizationFlags frame_flags = per_frame_flags;
        bool fix_cam1_extrinsics = false;
        bool fix_cam2_extrinsics = false;
        
        if (num_cameras < 2) {
            // Only one camera: don't optimize extrinsics
            frame_flags.optimize_inter_camera = false;
        } else if (!has_cam0 && has_cam1 && has_cam2) {
            // Only cam1 and cam2: fix cam1→cam0, optimize cam2→cam0
            frame_flags.optimize_inter_camera = true;
            fix_cam1_extrinsics = true;  // Fix cam1, optimize cam2
        } else {
            // Multiple cameras including cam0: optimize extrinsics normally
            frame_flags.optimize_inter_camera = true;
        }
        
        // Create local copies of intrinsics and extrinsics for this frame
        double frame_intrinsic_0[4], local_frame_dist_0[4];
        double frame_intrinsic_1[4], local_frame_dist_1[4];
        double frame_intrinsic_2[4], local_frame_dist_2[4];
        double frame_qvec_cam_1[4], frame_tvec_cam_1[3];
        double frame_qvec_cam_2[4], frame_tvec_cam_2[3];
        
        // Initialize from global values
        std::copy(intrinsic_0, intrinsic_0 + 4, frame_intrinsic_0);
        std::copy(dist_0, dist_0 + 4, local_frame_dist_0);
        std::copy(intrinsic_1, intrinsic_1 + 4, frame_intrinsic_1);
        std::copy(dist_1, dist_1 + 4, local_frame_dist_1);
        std::copy(intrinsic_2, intrinsic_2 + 4, frame_intrinsic_2);
        std::copy(dist_2, dist_2 + 4, local_frame_dist_2);
        std::copy(qvec_cam_1, qvec_cam_1 + 4, frame_qvec_cam_1);
        std::copy(tvec_cam_1, tvec_cam_1 + 3, frame_tvec_cam_1);
        std::copy(qvec_cam_2, qvec_cam_2 + 4, frame_qvec_cam_2);
        std::copy(tvec_cam_2, tvec_cam_2 + 3, frame_tvec_cam_2);

        //Save a calibration file before each optimization, with appropriate name
        std::string filename = "calibration_result_initial_frame_" + std::to_string(i) + ".json";
        SaveCalibrationResult(filename,
            intrinsic_0, dist_0,
            intrinsic_1, dist_1,
            intrinsic_2, dist_2,
            qvec_cam_1, tvec_cam_1,
            qvec_cam_2, tvec_cam_2,
            frame_target_poses, single_timestamp);
        
        // Optimize this frame's parameters
        OptimizeFishEyeParameters(
            frame_intrinsic_0, local_frame_dist_0,
            img_pts_list_0_frame, obj_pts_list_0_frame,
            frame_intrinsic_1, local_frame_dist_1,
            img_pts_list_1_frame, obj_pts_list_1_frame,
            frame_intrinsic_2, local_frame_dist_2,
            img_pts_list_2_frame, obj_pts_list_2_frame,
            frame_qvec_cam_1, frame_tvec_cam_1,
            frame_qvec_cam_2, frame_tvec_cam_2,
            frame_target_poses,  // local pose only
            single_timestamp,
            frame_flags,
            fix_cam1_extrinsics,
            fix_cam2_extrinsics
        );
        
        // Update global target poses
        target_poses[i] = frame_target_poses[0];
        
        // Store optimized intrinsics for averaging
        if (has_cam0) {
            frame_intrinsics_0.push_back({frame_intrinsic_0[0], frame_intrinsic_0[1], 
                                         frame_intrinsic_0[2], frame_intrinsic_0[3]});
            frame_dist_0.push_back({local_frame_dist_0[0], local_frame_dist_0[1], 
                                   local_frame_dist_0[2], local_frame_dist_0[3]});
        }
        if (has_cam1) {
            frame_intrinsics_1.push_back({frame_intrinsic_1[0], frame_intrinsic_1[1], 
                                         frame_intrinsic_1[2], frame_intrinsic_1[3]});
            frame_dist_1.push_back({local_frame_dist_1[0], local_frame_dist_1[1], 
                                   local_frame_dist_1[2], local_frame_dist_1[3]});
        }
        if (has_cam2) {
            frame_intrinsics_2.push_back({frame_intrinsic_2[0], frame_intrinsic_2[1], 
                                         frame_intrinsic_2[2], frame_intrinsic_2[3]});
            frame_dist_2.push_back({local_frame_dist_2[0], local_frame_dist_2[1], 
                                   local_frame_dist_2[2], local_frame_dist_2[3]});
        }
        
        // Store optimized extrinsics for averaging based on camera visibility
        if (has_cam0 && has_cam1) {
            // Direct cam1→cam0 estimate
            Eigen::Matrix4d T_cam1_in_cam0 = quatTransToMatrix(frame_qvec_cam_1, frame_tvec_cam_1);
            cam1_to_cam0_list.push_back(T_cam1_in_cam0);
        }
        
        if (has_cam0 && has_cam2) {
            // Direct cam2→cam0 estimate
            Eigen::Matrix4d T_cam2_in_cam0 = quatTransToMatrix(frame_qvec_cam_2, frame_tvec_cam_2);
            cam2_to_cam0_list.push_back(T_cam2_in_cam0);
        } else if (has_cam1 && has_cam2) {
            // Indirect cam2→cam0 via cam1→cam0
            // frame_qvec_cam_1 is fixed (cam1→cam0), so use the value we started with (global)
            Eigen::Matrix4d T_cam1_in_cam0;
            if (fix_cam1_extrinsics) {
                // cam1 was fixed, use the value we started with (global)
                T_cam1_in_cam0 = quatTransToMatrix(qvec_cam_1, tvec_cam_1);
            } else {
                // cam1 was optimized, use optimized value (shouldn't happen if only cam1 and cam2)
                T_cam1_in_cam0 = quatTransToMatrix(frame_qvec_cam_1, frame_tvec_cam_1);
            }
            
            // frame_qvec_cam_2 is cam2→cam0 (optimized in this frame)
            Eigen::Matrix4d T_cam2_in_cam0_direct = quatTransToMatrix(frame_qvec_cam_2, frame_tvec_cam_2);
            
            // This is already cam2→cam0, so we can use it directly
            cam2_to_cam0_list.push_back(T_cam2_in_cam0_direct);
        }
    }
    
    // Average intrinsics across frames for each camera
    std::cout << "Averaging intrinsics across frames..." << std::endl;
    
    if (!frame_intrinsics_0.empty()) {
        std::array<double, 4> avg_intrinsic_0 = {0, 0, 0, 0};
        std::array<double, 4> avg_dist_0 = {0, 0, 0, 0};
        for (const auto& intr : frame_intrinsics_0) {
            for (int j = 0; j < 4; ++j) avg_intrinsic_0[j] += intr[j];
        }
        for (const auto& dist : frame_dist_0) {
            for (int j = 0; j < 4; ++j) avg_dist_0[j] += dist[j];
        }
        double n = frame_intrinsics_0.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_0[j] = avg_intrinsic_0[j] / n;
            dist_0[j] = avg_dist_0[j] / n;
        }
        std::cout << "Camera 0: Averaged " << n << " frame estimates" << std::endl;
    }
    
    if (!frame_intrinsics_1.empty()) {
        std::array<double, 4> avg_intrinsic_1 = {0, 0, 0, 0};
        std::array<double, 4> avg_dist_1 = {0, 0, 0, 0};
        for (const auto& intr : frame_intrinsics_1) {
            for (int j = 0; j < 4; ++j) avg_intrinsic_1[j] += intr[j];
        }
        for (const auto& dist : frame_dist_1) {
            for (int j = 0; j < 4; ++j) avg_dist_1[j] += dist[j];
        }
        double n = frame_intrinsics_1.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_1[j] = avg_intrinsic_1[j] / n;
            dist_1[j] = avg_dist_1[j] / n;
        }
        std::cout << "Camera 1: Averaged " << n << " frame estimates" << std::endl;
    }
    
    if (!frame_intrinsics_2.empty()) {
        std::array<double, 4> avg_intrinsic_2 = {0, 0, 0, 0};
        std::array<double, 4> avg_dist_2 = {0, 0, 0, 0};
        for (const auto& intr : frame_intrinsics_2) {
            for (int j = 0; j < 4; ++j) avg_intrinsic_2[j] += intr[j];
        }
        for (const auto& dist : frame_dist_2) {
            for (int j = 0; j < 4; ++j) avg_dist_2[j] += dist[j];
        }
        double n = frame_intrinsics_2.size();
        for (int j = 0; j < 4; ++j) {
            intrinsic_2[j] = avg_intrinsic_2[j] / n;
            dist_2[j] = avg_dist_2[j] / n;
        }
        std::cout << "Camera 2: Averaged " << n << " frame estimates" << std::endl;
    }
    
    // Average extrinsics using hierarchical approach
    std::cout << "Averaging extrinsics across frames..." << std::endl;
    
    // Fix cam0, average cam1→cam0
    if (!cam1_to_cam0_list.empty()) {
        Eigen::Matrix4d cam1_in_cam0_avg = averagePoses(cam1_to_cam0_list);
        Eigen::Quaterniond q1(cam1_in_cam0_avg.block<3,3>(0,0));
        qvec_cam_1[0] = q1.w(); qvec_cam_1[1] = q1.x(); qvec_cam_1[2] = q1.y(); qvec_cam_1[3] = q1.z();
        tvec_cam_1[0] = cam1_in_cam0_avg(0,3);
        tvec_cam_1[1] = cam1_in_cam0_avg(1,3);
        tvec_cam_1[2] = cam1_in_cam0_avg(2,3);
        std::cout << "Camera 1→Camera 0: Averaged " << cam1_to_cam0_list.size() << " frame estimates" << std::endl;
    }
    
    // Average cam2→cam0 (direct and indirect)
    if (!cam2_to_cam0_list.empty()) {
        Eigen::Matrix4d cam2_in_cam0_avg = averagePoses(cam2_to_cam0_list);
        Eigen::Quaterniond q2(cam2_in_cam0_avg.block<3,3>(0,0));
        qvec_cam_2[0] = q2.w(); qvec_cam_2[1] = q2.x(); qvec_cam_2[2] = q2.y(); qvec_cam_2[3] = q2.z();
        tvec_cam_2[0] = cam2_in_cam0_avg(0,3);
        tvec_cam_2[1] = cam2_in_cam0_avg(1,3);
        tvec_cam_2[2] = cam2_in_cam0_avg(2,3);
        std::cout << "Camera 2→Camera 0: Averaged " << cam2_to_cam0_list.size() << " frame estimates" << std::endl;
    }
    

    SaveCalibrationResult("/home/jake/calibration_w_eigen/calibration_post_processing.json",
        intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses, master_timestamps
    );
    // std::cin.get();

    std::cout << "Stage 2: Global optimization..." << std::endl;

    OptimizationFlags global_flags;
    // Load global flags from file if provided, otherwise use defaults
    bool global_flags_loaded = false;
    if (!global_flags_file.empty()) {
        global_flags_loaded = LoadOptimizationFlagsFromJson(global_flags_file, global_flags);
        if (!global_flags_loaded) {
            std::cout << "Warning: Failed to load global flags from file, using defaults." << std::endl;
        }
    }
    
    if (!global_flags_loaded) {
        // Default global flags
        global_flags.optimize_intrinsics = false;
        global_flags.optimize_distortion = true;
        global_flags.optimize_inter_camera = true;
        global_flags.optimize_target_poses = true;
        // std::cout << "Using default global optimization flags:" << std::endl;
        // std::cout << "  optimize_intrinsics: " << global_flags.optimize_intrinsics << std::endl;
        // std::cout << "  optimize_distortion: " << global_flags.optimize_distortion << std::endl;
        // std::cout << "  optimize_inter_camera: " << global_flags.optimize_inter_camera << std::endl;
        // std::cout << "  optimize_target_poses: " << global_flags.optimize_target_poses << std::endl;
    }


    // Step 7: Optimize fisheye parameters
    OptimizeFishEyeParameters(
        intrinsic_0, dist_0,
        img_pts_list_0, obj_pts_list_0,
        intrinsic_1, dist_1,
        img_pts_list_1, obj_pts_list_1,
        intrinsic_2, dist_2,
        img_pts_list_2, obj_pts_list_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses,
        master_timestamps,
        global_flags
    );


    std::vector<double> timestamps_0_d(filtered_timestamp_list_0.begin(), filtered_timestamp_list_0.end());
    std::vector<double> timestamps_1_d(filtered_timestamp_list_1.begin(), filtered_timestamp_list_1.end());
    std::vector<double> timestamps_2_d(filtered_timestamp_list_2.begin(), filtered_timestamp_list_2.end());

    
    SaveCalibrationResult("/home/jake/calibration_w_eigen/calibration_result_final.json",
        intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses, master_timestamps
    );

    std::copy(intrinsic_0, intrinsic_0 + 4, intrinsic_2);  // Copy 4 elements
    std::copy(dist_0, dist_0 + 4, dist_2);  // Copy 4 elements

    // ========== COMMENTED OUT: Visualization code using old extrinsics (not available in new initialization) ==========
    // auto cam0_data = GenerateReprojectionErrorData(
    //     intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0, filtered_timestamp_list_0);
    // 
    // auto cam1_data = GenerateReprojectionErrorData(
    //     intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1, filtered_timestamp_list_1);
    //
    // auto cam2_data = GenerateReprojectionErrorData(
    //     intrinsic_2, dist_2, extrinsics_2, img_pts_list_2, obj_pts_list_2, filtered_timestamp_list_2);
    // ========== END COMMENTED OUT VISUALIZATION CODE ==========

    
    // Output refined parameters
    std::cout << "Refined Intrinsic Parameters for Camera 0:\n";
    std::cout << "fx: " << intrinsic_0[0] << ", fy: " << intrinsic_0[1]
              << ", cx: " << intrinsic_0[2] << ", cy: " << intrinsic_0[3] << std::endl;
    std::cout << "Distortion Coefficients for Camera 0: ";
    for (double d : dist_0) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << "Refined Intrinsic Parameters for Camera 1:\n";
    std::cout << "fx: " << intrinsic_1[0] << ", fy: " << intrinsic_1[1]
              << ", cx: " << intrinsic_1[2] << ", cy: " << intrinsic_1[3] << std::endl;
    std::cout << "Distortion Coefficients for Camera 1: ";
    for (double d : dist_1) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << "Refined Intrinsic Parameters for Camera 2:\n";
    std::cout << "fx: " << intrinsic_2[0] << ", fy: " << intrinsic_2[1]
              << ", cx: " << intrinsic_2[2] << ", cy: " << intrinsic_2[3] << std::endl;
    std::cout << "Distortion Coefficients for Camera 2: ";
    for (double d : dist_2) std::cout << d << " ";
    std::cout << std::endl;

    ceres::QuaternionToAngleAxis(qvec_cam_1, rvec_cam_1);
    std::cout << "Inter-camera Rotation Vector (Camera 1): ";
    for (double r : rvec_cam_1) std::cout << r << " ";
    std::cout << "\nInter-camera Translation Vector (Camera 1): ";
    for (double t : tvec_cam_1) std::cout << t << " ";
    std::cout << std::endl;

    ceres::QuaternionToAngleAxis(qvec_cam_2, rvec_cam_2);
    std::cout << "Inter-camera Rotation Vector (Camera 2): ";
    for (double r : rvec_cam_2) std::cout << r << " ";
    std::cout << "\nInter-camera Translation Vector (Camera 2): ";
    for (double t : tvec_cam_2) std::cout << t << " ";
    std::cout << std::endl;

    return 0;
}



























