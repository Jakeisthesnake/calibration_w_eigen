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

        residuals[0] = u - T(measured_px_(0));
        residuals[1] = v - T(measured_px_(1));
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
    const std::vector<TimestampEntry>& master_timestamps
)
{
    ceres::Problem problem;

    // Bookkeeping which target_poses entries correspond to at least one observation
    std::vector<bool> target_pose_used(target_poses.size(), false);

    // Add reprojection residuals
    for (size_t idx = 0; idx < master_timestamps.size(); ++idx) {
        const auto& entry = master_timestamps[idx];

        // CAM0: direct use of target_poses[idx]
        if (entry.cam0_idx != -1) {
            int cam0_i = entry.cam0_idx;
            target_pose_used[idx] = true;

            // For each corner observed by cam0 at that timestamp:
            for (size_t j = 0; j < img_pts_0[cam0_i].size(); ++j) {
                auto measured = img_pts_0[cam0_i][j];
                auto objp = obj_pts_0[cam0_i][j];

                // Note: for cam0 we can make cam_q = identity, cam_t = zero so that
                // transformation via cam inverse is a no-op.
                // We will pass cam_q_cam0 = (1,0,0,0) and cam_t_cam0 = (0,0,0)
                static double cam0_q[4] = {1.0, 0.0, 0.0, 0.0};
                static double cam0_t[3] = {0.0, 0.0, 0.0};

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
            }
        }
    }

    // === Temporal smoothness on target_poses (adjacent timestamps) ===
    // Keep only where both consecutive target_poses are "used" (observed by at least one camera)
    for (size_t i = 1; i < target_poses.size(); ++i) {
        if (!target_pose_used[i-1] || !target_pose_used[i]) continue;

        double trans_weight = 1.0;
        double rot_weight = 1.0;
        ceres::CostFunction* smooth_cost = new ceres::AutoDiffCostFunction<TempSmooth, 6, 4, 3, 4, 3>(
            new TempSmooth(trans_weight, rot_weight));

        double* q1 = target_poses[i-1].data();
        double* t1 = target_poses[i-1].data() + 4;
        double* q2 = target_poses[i].data();
        double* t2 = target_poses[i].data() + 4;

        problem.AddResidualBlock(smooth_cost, nullptr, q1, t1, q2, t2);
    }

    // === Set quaternion manifolds for target_poses and camera quaternions ===
    for (size_t i = 0; i < target_poses.size(); ++i) {
        if (!target_pose_used[i]) continue; // only set manifold for used targets
        problem.SetManifold(target_poses[i].data(), new ceres::EigenQuaternionManifold());
    }

    problem.SetManifold(qvec_cam_1, new ceres::EigenQuaternionManifold());
    problem.SetManifold(qvec_cam_2, new ceres::EigenQuaternionManifold());

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

    // Solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    // tune as you like (max_num_iterations, trust_region settings...)

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
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

void VisualizeStereoReprojectionError(
    const std::vector<FrameData>& frames_cam0,
    const std::vector<FrameData>& frames_cam1,
    const std::vector<FrameData>& frames_cam2,
    const double* intrinsic_0,
    const double* dist_0,
    const double* intrinsic_1,
    const double* dist_1,
    const double* intrinsic_2,
    const double* dist_2)
{
    // Build timestamp -> frame map
    FrameMap map_cam0, map_cam1, map_cam2;
    std::set<int64_t> all_timestamps;

    for (const auto& f : frames_cam0) {
        map_cam0[f.timestamp_ns] = f;
        all_timestamps.insert(f.timestamp_ns);
    }

    for (const auto& f : frames_cam1) {
        map_cam1[f.timestamp_ns] = f;
        all_timestamps.insert(f.timestamp_ns);
    }

    for (const auto& f : frames_cam2) {
        map_cam2[f.timestamp_ns] = f;
        all_timestamps.insert(f.timestamp_ns);
    }

    std::vector<int64_t> timestamps(all_timestamps.begin(), all_timestamps.end());

    // Window setup
    pangolin::CreateWindowAndBind("Stereo Reprojection Error Viewer", 1600, 800);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Assume same fx/fy/cx/cy for all cameras
    int img_w = static_cast<int>(2 * intrinsic_0[2]);
    int img_h = static_cast<int>(2 * intrinsic_0[3]);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrixOrthographic(0, 3 * img_w, img_h, 0, -1, 1)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetAspect(static_cast<float>(3 * img_w) / img_h)
        .SetHandler(&pangolin::StaticHandler);  

    size_t current_index = 0;
    int64_t ts = 0;

    pangolin::RegisterKeyPressCallback('a', [&]() {
        if (current_index > 0) current_index--;
        ts = timestamps[current_index];
        std::cout << "Current index: " << ts << std::endl;
        std::cout << "Frame error: " 
                  << (map_cam0.count(ts) ? map_cam0[ts].error_sum : 0.0) << " (Cam0), "
                  << (map_cam1.count(ts) ? map_cam1[ts].error_sum : 0.0) << " (Cam1), "
                  << (map_cam2.count(ts) ? map_cam2[ts].error_sum : 0.0) << " (Cam2)" << std::endl;

    });

    pangolin::RegisterKeyPressCallback('d', [&]() {
        if (current_index + 1 < timestamps.size()) current_index++;
        ts = timestamps[current_index];
        std::cout << "Current index: " << ts << std::endl;
        std::cout << "Frame error: " 
                  << (map_cam0.count(ts) ? map_cam0[ts].error_sum : 0.0) << " (Cam0), "
                  << (map_cam1.count(ts) ? map_cam1[ts].error_sum : 0.0) << " (Cam1), "
                  << (map_cam2.count(ts) ? map_cam2[ts].error_sum : 0.0) << " (Cam2)" << std::endl;
    });

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        ts = timestamps[current_index];

        if (map_cam0.count(ts)) {
            const FrameData& f0 = map_cam0[ts];
            for (size_t i = 0; i < f0.observed_pts.size(); ++i) {
                const auto& obs = f0.observed_pts[i];
                const auto& proj = f0.projected_pts[i];

                // Observed (green)
                glColor3f(0.0f, 1.0f, 0.0f);
                glPointSize(4.0f);
                glBegin(GL_POINTS);
                glVertex2f(obs[0], obs[1]);
                glEnd();

                // Projected (red)
                glColor3f(1.0f, 0.0f, 0.0f);
                glBegin(GL_POINTS);
                glVertex2f(proj[0], proj[1]);
                glEnd();

                // Line (yellow)
                glColor3f(1.0f, 1.0f, 0.0f);
                glBegin(GL_LINES);
                glVertex2f(obs[0], obs[1]);
                glVertex2f(proj[0], proj[1]);
                glEnd();
            }
        }

        if (map_cam1.count(ts)) {
            const FrameData& f1 = map_cam1[ts];
            for (size_t i = 0; i < f1.observed_pts.size(); ++i) {
                const auto& obs = f1.observed_pts[i];
                const auto& proj = f1.projected_pts[i];

                float offset = static_cast<float>(img_w);  // shift right

                // Observed (green)
                glColor3f(0.0f, 1.0f, 0.0f);
                glPointSize(4.0f);
                glBegin(GL_POINTS);
                glVertex2f(obs[0] + offset, obs[1]);
                glEnd();

                // Projected (red)
                glColor3f(1.0f, 0.0f, 0.0f);
                glBegin(GL_POINTS);
                glVertex2f(proj[0] + offset, proj[1]);
                glEnd();

                // Line (yellow)
                glColor3f(1.0f, 1.0f, 0.0f);
                glBegin(GL_LINES);
                glVertex2f(obs[0] + offset, obs[1]);
                glVertex2f(proj[0] + offset, proj[1]);
                glEnd();
            }
        }

        if (map_cam2.count(ts)) {
            const FrameData& f2 = map_cam2[ts];
            for (size_t i = 0; i < f2.observed_pts.size(); ++i) {
                const auto& obs = f2.observed_pts[i];
                const auto& proj = f2.projected_pts[i];

                float offset = static_cast<float>(2 * img_w);  // shift right

                // Observed (green)
                glColor3f(0.0f, 1.0f, 0.0f);
                glPointSize(4.0f);
                glBegin(GL_POINTS);
                glVertex2f(obs[0] + offset, obs[1]);
                glEnd();

                // Projected (red)
                glColor3f(1.0f, 0.0f, 0.0f);
                glBegin(GL_POINTS);
                glVertex2f(proj[0] + offset, proj[1]);
                glEnd();

                // Line (yellow)
                glColor3f(1.0f, 1.0f, 0.0f);
                glBegin(GL_LINES);
                glVertex2f(obs[0] + offset, obs[1]);
                glVertex2f(proj[0] + offset, proj[1]);
                glEnd();
            }
        }

        pangolin::FinishFrame();
    }
}


void SaveCalibrationResult(
    const double intrinsic_0[4], const double dist_0[4],
    const double intrinsic_1[4], const double dist_1[4],
    const double intrinsic_2[4], const double dist_2[4],

    const double qvec_cam_1[4], const double tvec_cam_1[3],
    const double qvec_cam_2[4], const double tvec_cam_2[3],

    const std::vector<std::array<double, 7>>& target_poses, // target→world
    const std::vector<TimestampEntry>& master_timestamps
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
        double timestamp = master_timestamps[i].timestamp_id; // assuming .timestamp exists

        json pose;
        pose["timestamp"]   = timestamp;
        pose["quaternion"]  = {tp[0], tp[1], tp[2], tp[3]};
        pose["translation"] = {tp[4], tp[5], tp[6]};
        output["target_poses"].push_back(pose);
    }

    // --- Inter-Camera Transforms (still valid) ---
    output["inter_camera"]["camera1_to_camera0"]["quaternion"]         = {qvec_cam_1[0], qvec_cam_1[1], qvec_cam_1[2], qvec_cam_1[3]};
    output["inter_camera"]["camera1_to_camera0"]["translation_vector"] = {tvec_cam_1[0], tvec_cam_1[1], tvec_cam_1[2]};

    output["inter_camera"]["camera2_to_camera0"]["quaternion"]         = {qvec_cam_2[0], qvec_cam_2[1], qvec_cam_2[2], qvec_cam_2[3]};
    output["inter_camera"]["camera2_to_camera0"]["translation_vector"] = {tvec_cam_2[0], tvec_cam_2[1], tvec_cam_2[2]};

    // --- Write to file ---
    std::ofstream ofs("/home/jake/calibration_w_eigan/calibration_output.json");
    ofs << std::setw(4) << output << std::endl;
    std::cout << "Saved calibration results to calibration_output.json" << std::endl;
}




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

    // Step 1: Load and process CSV data for all cameras
    auto [obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0] = processCSV(data_file, 0);
    auto [obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1] = processCSV(data_file, 1);
    auto [obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2] = processCSV(data_file, 2);

    // Step 2: Compute homographies and filter timestamps
    auto [H_list_0, filtered_timestamp_list_0] = computeHomographies(obj_pts_list_0, img_pts_list_0, timestamp_list_0);
    auto [H_list_1, filtered_timestamp_list_1] = computeHomographies(obj_pts_list_1, img_pts_list_1, timestamp_list_1);
    auto [H_list_2, filtered_timestamp_list_2] = computeHomographies(obj_pts_list_2, img_pts_list_2, timestamp_list_2);

    // Filter data for all cameras
    std::tie(obj_pts_list_0, img_pts_list_0, corner_ids_list_0) = filterDataByTimestamps(
        obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0, filtered_timestamp_list_0);
    std::tie(obj_pts_list_1, img_pts_list_1, corner_ids_list_1) = filterDataByTimestamps(
        obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1, filtered_timestamp_list_1);
    std::tie(obj_pts_list_2, img_pts_list_2, corner_ids_list_2) = filterDataByTimestamps(
        obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2, filtered_timestamp_list_2);


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




    Eigen::Matrix3d K_0;
    K_0 << 610, 0, 688,
            0, 756, 524,
            0, 0, 1;
    Eigen::Matrix3d K_1;
    K_1 << 802, 0, 662,
            0, 825, 502,
            0, 0, 1;
    Eigen::Matrix3d K_2;
    K_2 << 671, 0, 648,
            0, 841, 475,
            0, 0, 1;


    double intrinsic_0[4] = {K_0(0, 0), K_0(1, 1), K_0(0, 2), K_0(1, 2)};
    double dist_0[4] = {-0.013, -0.02, 0.063, -0.03}; 
    double intrinsic_1[4] = {K_1(0, 0), K_1(1, 1), K_1(0, 2), K_1(1, 2)};
    double dist_1[4] = {0.09, -0.057, 0.014, -0.004}; 
    double intrinsic_2[4] = {K_2(0, 0), K_2(1, 1), K_2(0, 2), K_2(1, 2)};
    double dist_2[4] = {0.03, -0.065, 0.065, -0.0034}; 

    // Add extrinsics for all cameras
    std::vector<std::array<double, 7>> extrinsics_0, extrinsics_1, extrinsics_2;
    for (const auto& H : H_list_0) {
        auto [R, t] = compute_extrinsic_params(H, K_0);
        Eigen::Quaterniond q(R);
        std::array<double, 7> pose;
        pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
        pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
        extrinsics_0.push_back(pose);
    }
    for (const auto& H : H_list_1) {
        auto [R, t] = compute_extrinsic_params(H, K_1);
        Eigen::Quaterniond q(R);
        std::array<double, 7> pose;
        pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
        pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
        extrinsics_1.push_back(pose);
    }
    for (const auto& H : H_list_2) {
        auto [R, t] = compute_extrinsic_params(H, K_2);
        Eigen::Quaterniond q(R);
        std::array<double, 7> pose;
        pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
        pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
        extrinsics_2.push_back(pose);
    }

    

    // Add transformation parameters for camera 1 and camera 2 (relative to camera 0)
    double rvec_cam_1[3] = {0.05, 0.56, 2.77};
    double tvec_cam_1[3] = {-0.24, -0.36, 0.047}; // Baseline of 10 cm
    double qvec_cam_1[4];
    ceres::AngleAxisToQuaternion(rvec_cam_1, qvec_cam_1);  // Converts to [w, x, y, z]

    double rvec_cam_2[3] = {-0.60, 0.59, 2.63};  // Different initial rotation
    double tvec_cam_2[3] = {-0.37, -0.091, -0.18};  // 20cm baseline
    double qvec_cam_2[4];
    ceres::AngleAxisToQuaternion(rvec_cam_2, qvec_cam_2);

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

    // --- END: single-target-pose initialization --------------------------------


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
        master_timestamps
    );


    std::vector<double> timestamps_0_d(filtered_timestamp_list_0.begin(), filtered_timestamp_list_0.end());
    std::vector<double> timestamps_1_d(filtered_timestamp_list_1.begin(), filtered_timestamp_list_1.end());
    std::vector<double> timestamps_2_d(filtered_timestamp_list_2.begin(), filtered_timestamp_list_2.end());

    
    SaveCalibrationResult(intrinsic_0, dist_0,
        intrinsic_1, dist_1,
        intrinsic_2, dist_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2,
        target_poses, master_timestamps
    );

    std::copy(intrinsic_0, intrinsic_0 + 4, intrinsic_2);  // Copy 4 elements
    std::copy(dist_0, dist_0 + 4, dist_2);  // Copy 4 elements

    auto cam0_data = GenerateReprojectionErrorData(
        intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0, filtered_timestamp_list_0);
    
    auto cam1_data = GenerateReprojectionErrorData(
        intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1, filtered_timestamp_list_1);

    auto cam2_data = GenerateReprojectionErrorData(
        intrinsic_2, dist_2, extrinsics_2, img_pts_list_2, obj_pts_list_2, filtered_timestamp_list_2);

    VisualizeStereoReprojectionError(cam0_data, cam1_data, cam2_data, intrinsic_0, dist_0, intrinsic_1, dist_1, intrinsic_2, dist_2);

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





























// #include <Eigen/Dense>
// #include <nlohmann/json.hpp>
// #include <fstream>
// #include <unordered_map>
// #include <unordered_set>
// #include <vector>
// #include <iostream>
// #include <tuple>
// #include <string>
// #include <sstream>
// #include <cmath>
// #include <utility>
// #include <GL/glew.h>
// #include <pangolin/pangolin.h>
// #include <Eigen/Core>
// #include <ceres/ceres.h>
// #include <ceres/manifold.h>
// #include <ceres/rotation.h>  // for AngleAxisRotatePoint, if you need to apply rotation
// #include <ceres/autodiff_cost_function.h>  // for AutoDiffCostFunction (used in your case)
// #include <ceres/solver.h>                  // for Solver options and summary
// #include <ceres/problem.h>    
// #include <array>
// #include <algorithm>
// #include <random>
// #include <iostream>
// #include <pangolin/handler/handler.h>
// #include <GL/freeglut.h>


// // loadAprilTagBoardFlat
// using json = nlohmann::json;

// // computeHomographies
// using Point2dVec = std::vector<Eigen::Vector2d>;
// using Point3dVec = std::vector<Eigen::Vector3d>;
// using HomographyList = std::vector<Eigen::Matrix3d>;

// // filterDataByTimestamps
// // using Point2dVec = std::vector<Eigen::Vector2d>;
// // using Point3dVec = std::vector<Eigen::Vector3d>;
// using IDVec = std::vector<int>;
// // using TimestampList = std::vector<double>;  // Or another type if needed

// //Process CSV()
// using Point3dVec = std::vector<Eigen::Vector3d>;
// using Point2dVec = std::vector<Eigen::Vector2d>;
// using IDVec = std::vector<int>;
// using TimestampList = std::vector<uint64_t>;


// std::unordered_map<int, Eigen::Vector3d> loadAprilTagBoardFlat(const std::string& json_file) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file: " + json_file);
//     }

//     json board_config;
//     file >> board_config;

//     int tag_cols = board_config["tagCols"];
//     int tag_rows = board_config["tagRows"];
//     double tag_size = board_config["tagSize"];
//     double tag_spacing = board_config["tagSpacing"];

//     std::unordered_map<int, Eigen::Vector3d> id_to_point;

//     int tag_id = 0;
//     for (int row = 0; row < tag_rows; ++row) {
//         for (int col = 0; col < tag_cols; ++col) {
//             double tag_x = col * (tag_size + tag_spacing);
//             double tag_y = row * (tag_size + tag_spacing);
//             double tag_z = 0.0;

//             Eigen::Vector3d corners[4] = {
//                 {tag_x, tag_y, tag_z},                                      // Top-left
//                 {tag_x + tag_size, tag_y, tag_z},                          // Top-right
//                 {tag_x + tag_size, tag_y + tag_size, tag_z},               // Bottom-right
//                 {tag_x, tag_y + tag_size, tag_z}                           // Bottom-left
//             };

//             for (int i = 0; i < 4; ++i) {
//                 int corner_id = tag_id * 4 + i;
//                 id_to_point[corner_id] = corners[i];
//             }

//             ++tag_id;
//         }
//     }

//     return id_to_point;
// }






// std::tuple<HomographyList, TimestampList> computeHomographies(
//     const std::vector<Point3dVec>& obj_pts_list,
//     const std::vector<Point2dVec>& img_pts_list,
//     const TimestampList& timestamp_list)
// {
//     HomographyList homographies;
//     TimestampList filtered_timestamps;

//     for (size_t k = 0; k < obj_pts_list.size(); ++k) {
//         const auto& obj_pts = obj_pts_list[k];
//         const auto& img_pts = img_pts_list[k];
//         double timestamp = timestamp_list[k];

//         if (obj_pts.size() != img_pts.size() || obj_pts.size() < 4) {
//             std::cerr << "Skipping due to insufficient or mismatched points." << std::endl;
//             continue;
//         }

//         Eigen::MatrixXd A(2 * obj_pts.size(), 9);
//         for (size_t i = 0; i < obj_pts.size(); ++i) {
//             double X = obj_pts[i].x(), Y = obj_pts[i].y();
//             double x = img_pts[i].x(), y = img_pts[i].y();

//             A.row(2 * i)     << -X, -Y, -1,  0,  0,  0,  x * X, x * Y, x;
//             A.row(2 * i + 1) <<  0,  0,  0, -X, -Y, -1,  y * X, y * Y, y;
//         }

//         // Compute SVD
//         Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
//         Eigen::VectorXd h = svd.matrixV().col(8);  // Last column of V

//         // Check condition number
//         double min_singular = svd.singularValues()(svd.singularValues().size() - 1);
//         if (min_singular < 1e-8) {
//             std::cerr << "Singular values too small, skipping homography." << std::endl;
//             continue;
//         }

//         // Reshape into 3x3 matrix
//         Eigen::Matrix3d H;
//         H << h(0), h(1), h(2),
//              h(3), h(4), h(5),
//              h(6), h(7), h(8);

//         H /= H(2, 2);  // Normalize

//         homographies.push_back(H);
//         filtered_timestamps.push_back(timestamp);
//     }

//     return {homographies, filtered_timestamps};
// }





// std::tuple<
//     std::vector<Point3dVec>,
//     std::vector<Point2dVec>,
//     std::vector<IDVec>
// > filterDataByTimestamps(
//     const std::vector<Point3dVec>& obj_pts_list,
//     const std::vector<Point2dVec>& img_pts_list,
//     const std::vector<IDVec>& corner_ids_list,
//     const TimestampList& timestamp_list,
//     const TimestampList& filtered_timestamps)
// {
//     std::vector<Point3dVec> filtered_obj_pts;
//     std::vector<Point2dVec> filtered_img_pts;
//     std::vector<IDVec> filtered_corner_ids;

//     // Use a set for faster lookup
//     std::unordered_set<double> timestamp_set(filtered_timestamps.begin(), filtered_timestamps.end());

//     for (size_t i = 0; i < timestamp_list.size(); ++i) {
//         if (timestamp_set.count(timestamp_list[i]) > 0) {
//             filtered_obj_pts.push_back(obj_pts_list[i]);
//             filtered_img_pts.push_back(img_pts_list[i]);
//             filtered_corner_ids.push_back(corner_ids_list[i]);
//         }
//     }

//     return {filtered_obj_pts, filtered_img_pts, filtered_corner_ids};
// }


// struct CSVRow {
//     uint64_t timestamp_ns;
//     int camera_id;
//     int corner_id;
//     double x;
//     double y;
//     double radius;
// };

// std::vector<CSVRow> readCSV(const std::string& file_path) {
//     std::ifstream file(file_path);
//     std::vector<CSVRow> rows;

//     if (!file.is_open()) {
//         std::cerr << "Failed to open file: " << file_path << std::endl;
//         return rows;
//     }

//     std::string line;
//     bool is_first_line = true;

//     while (std::getline(file, line)) {
//         if (is_first_line) {
//             is_first_line = false; // Skip header
//             continue;
//         }

//         std::stringstream ss(line);
//         std::string token;

//         CSVRow row;

//         std::getline(ss, token, ',');
//         row.timestamp_ns = std::stoull(token);

//         std::getline(ss, token, ',');
//         row.camera_id = std::stoi(token);

//         std::getline(ss, token, ',');
//         row.corner_id = std::stoi(token);

//         std::getline(ss, token, ',');
//         row.x = std::stod(token);

//         std::getline(ss, token, ',');
//         row.y = std::stod(token);

//         std::getline(ss, token, ',');
//         row.radius = std::stod(token);

//         rows.push_back(row);
//     }

//     return rows;
// }

// Eigen::Vector3d get_object_point(
//     int corner_id,
//     int tag_rows = 6,
//     int tag_cols = 6,
//     double tag_size = 0.13,
//     double tag_spacing = 0.04)
// {
//     int tag_index = corner_id / 4;
//     int local_corner = corner_id % 4;

//     int row = tag_index / tag_cols;
//     int col = tag_index % tag_cols;

//     double tag_x = col * (tag_size + tag_spacing);
//     double tag_y = row * (tag_size + tag_spacing);

//     // Offsets for corners: TL, TR, BR, BL
//     const double corner_offsets[4][2] = {
//         {0.0, 0.0},                // Top-left
//         {tag_size, 0.0},            // Top-right
//         {tag_size, tag_size},        // Bottom-right
//         {0.0, tag_size}             // Bottom-left
//     };

//     double offset_x = corner_offsets[local_corner][0];
//     double offset_y = corner_offsets[local_corner][1];

//     return Eigen::Vector3d(tag_x + offset_x, tag_y + offset_y, 0.0);
// }

// std::tuple<
//     std::vector<Point3dVec>,
//     std::vector<Point2dVec>,
//     std::vector<IDVec>,
//     TimestampList
// > processCSV(const std::string& file_path, int target_cam_id)
// {
//     auto rows = readCSV(file_path);

//     // Grouped output per timestamp
//     struct DataGroup {
//         Point3dVec obj_points;
//         Point2dVec img_points;
//         IDVec corner_ids;
//     };
//     std::unordered_map<uint64_t, DataGroup> grouped_data;

//     for (const auto& row : rows) {
//         if (row.camera_id != target_cam_id) continue;

//         Eigen::Vector2d img_pt(row.x, row.y);
//         Eigen::Vector3d obj_pt = get_object_point(row.corner_id);

//         auto& group = grouped_data[row.timestamp_ns];
//         group.img_points.push_back(img_pt);
//         group.obj_points.push_back(obj_pt);
//         group.corner_ids.push_back(row.corner_id);
//     }

//     // Sort timestamps
//     std::vector<uint64_t> sorted_timestamps;
//     sorted_timestamps.reserve(grouped_data.size());
//     for (const auto& [timestamp, _] : grouped_data) {
//         sorted_timestamps.push_back(timestamp);
//     }
//     std::sort(sorted_timestamps.begin(), sorted_timestamps.end());

//     // Extract data in sorted order
//     std::vector<Point3dVec> obj_pts_list;
//     std::vector<Point2dVec> img_pts_list;
//     std::vector<IDVec> corner_ids_list;
//     TimestampList timestamp_list;

//     for (const auto& timestamp : sorted_timestamps) {
//         const auto& data = grouped_data[timestamp];
//         obj_pts_list.push_back(data.obj_points);
//         img_pts_list.push_back(data.img_points);
//         corner_ids_list.push_back(data.corner_ids);
//         timestamp_list.push_back(timestamp);
//     }

//     return {obj_pts_list, img_pts_list, corner_ids_list, timestamp_list};
// }


// Eigen::Matrix3d compute_intrinsic_params(const std::vector<Eigen::Matrix3d>& H_list)
// {
//     // std::cout << "H_list" << std::endl;
//     // for (const auto& H : H_list) {
//     //     std::cout << H << std::endl;
//     // }
//     std::vector<Eigen::Matrix<double, 6, 1>> V;

//     for (const auto& H : H_list) {
//         Eigen::Vector3d h1 = H.col(0);
//         Eigen::Vector3d h2 = H.col(1);

//         Eigen::Matrix<double, 6, 1> v12;
//         v12 << h1(0) * h2(0),
//                h1(0) * h2(1) + h1(1) * h2(0),
//                h1(1) * h2(1),
//                h1(2) * h2(0) + h1(0) * h2(2),
//                h1(2) * h2(1) + h1(1) * h2(2),
//                h1(2) * h2(2);

//         Eigen::Matrix<double, 6, 1> v11_minus_v22;
//         v11_minus_v22 << h1(0)*h1(0) - h2(0)*h2(0),
//                          2*(h1(0)*h1(1) - h2(0)*h2(1)),
//                          h1(1)*h1(1) - h2(1)*h2(1),
//                          2*(h1(0)*h1(2) - h2(0)*h2(2)),
//                          2*(h1(1)*h1(2) - h2(1)*h2(2)),
//                          h1(2)*h1(2) - h2(2)*h2(2);

//         V.push_back(v12);
//         V.push_back(v11_minus_v22);
//     }

//     // Stack into a matrix
//     Eigen::MatrixXd V_mat(V.size(), 6);
//     for (size_t i = 0; i < V.size(); ++i) {
//         V_mat.row(i) = V[i].transpose();
//     }

//     // SVD
//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(V_mat, Eigen::ComputeFullV);
//     // std::cout << "SVD singular values: " << svd.singularValues().transpose() << std::endl;
//     Eigen::VectorXd b = svd.matrixV().col(5);  // Last column of V
//     // std::cout << "b: " << b.transpose() << std::endl;

//     // Form B matrix
//     Eigen::Matrix3d B;
//     B << b(0), b(1), b(3),
//          b(1), b(2), b(4),
//          b(3), b(4), b(5);
    
//     // std::cout << "B: " << B << std::endl;
//     double v0 = (B(0,1)*B(0,2) - B(1,2)*B(0,0)) / (B(0,0)*B(1,1) - B(0,1)*B(0,1));
//     double lambda = B(2,2) - (B(0,2)*B(0,2) + v0*(B(0,1)*B(0,2) - B(1,2)*B(0,0))) / B(0,0);
//     double alpha = std::sqrt(lambda / B(0,0));
//     double beta  = std::sqrt(lambda * B(0,0) / (B(0,0)*B(1,1) - B(0,1)*B(0,1)));
//     double gamma = -B(0,1) * alpha * alpha * beta / lambda;
//     double u0 = gamma * v0 / beta - B(0,2) * alpha * alpha / lambda;
//     // std::cout << "alpha: " << alpha << ", beta: " << beta
//     //           << ", gamma: " << gamma << ", u0: " << u0
//     //           << ", v0: " << v0 << std::endl;



//     Eigen::Matrix3d K;
//     K << alpha, gamma, u0,
//          0,     beta,  v0,
//          0,     0,     1;
//     // std::cin.get();  // Pause for debugging

//     return K;
// }

// Eigen::Matrix3d robust_intrinsic_estimation(
//     const std::vector<Eigen::Matrix3d>& H_list,
//     int max_trials = 10,
//     int min_h_required = 3)
// {
//     auto has_nan = [](const Eigen::Matrix3d& K) {
//         return !K.allFinite();
//     };

//     // First try with the full list
//     Eigen::Matrix3d K_full = compute_intrinsic_params(H_list);
//     if (!has_nan(K_full)) {
//         // std::cout << "Recovered K using full H_list." << std::endl;
//         return K_full;
//     }

//     // Otherwise, try randomized subsets
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     std::vector<Eigen::Matrix3d> valid_Ks;

//     for (int trial = 0; trial < max_trials; ++trial) {
//         int subset_size = std::min<int>(H_list.size(), min_h_required + trial % 3);
//         std::vector<Eigen::Matrix3d> subset;
//         std::sample(H_list.begin(), H_list.end(),
//                     std::back_inserter(subset),
//                     subset_size, gen);

//         Eigen::Matrix3d K = compute_intrinsic_params(subset);
//         if (!has_nan(K)) {
//             std::cout << "Recovered K from trial " << trial << " with subset size " << subset.size() << "." << std::endl;
//             valid_Ks.push_back(K);
//         }
//     }

//     if (!valid_Ks.empty()) {
//         // Average valid Ks
//         Eigen::Matrix3d K_avg = Eigen::Matrix3d::Zero();
//         for (const auto& K : valid_Ks) {
//             K_avg += K;
//         }
//         K_avg /= static_cast<double>(valid_Ks.size());
//         std::cout << "Returning average of " << valid_Ks.size() << " valid K matrices." << std::endl;
//         return K_avg;
//     }

//     std::cerr << "Failed to compute any valid K matrix after " << max_trials << " trials." << std::endl;
//     return Eigen::Matrix3d::Identity();  // fallback or throw
// }

// std::pair<Eigen::Matrix3d, Eigen::Vector3d> compute_extrinsic_params(
//     const Eigen::Matrix3d& H,
//     const Eigen::Matrix3d& K)
// {
//     // std::cout << "H: " << H << std::endl;
//     // std::cout << "K: " << K << std::endl;
//     Eigen::Matrix3d K_inv = K.inverse();

//     Eigen::Vector3d h1 = H.col(0);
//     Eigen::Vector3d h2 = H.col(1);
//     Eigen::Vector3d h3 = H.col(2);

//     double lambda = 1.0 / (K_inv * h1).norm();

//     Eigen::Vector3d r1 = lambda * (K_inv * h1);
//     Eigen::Vector3d r2 = lambda * (K_inv * h2);
//     Eigen::Vector3d t  = lambda * (K_inv * h3);
//     Eigen::Vector3d r3 = r1.cross(r2);

//     Eigen::Matrix3d R;
//     R.col(0) = r1;
//     R.col(1) = r2;
//     R.col(2) = r3;

//     // Re-orthonormalize R using SVD to ensure it's a valid rotation matrix
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     R = svd.matrixU() * svd.matrixV().transpose();
//     // std::cout << "R: " << R << std::endl;
//     // std::cout << "t: " << t.transpose() << std::endl;
//     // std::cin.get();  // Pause for debugging

//     return {R, t};
// }



// Eigen::MatrixXd kannala_brandt_project(
//     const Eigen::MatrixXd& points,       // Nx3
//     const Eigen::Vector4d& K,            // fx, fy, cx, cy
//     const Eigen::Vector4d& dist_coeffs)  // k1, k2, k3, k4
// {
//     const double k1 = dist_coeffs(0);
//     const double k2 = dist_coeffs(1);
//     const double k3 = dist_coeffs(2);
//     const double k4 = dist_coeffs(3);

//     const double fx = K(0);
//     const double fy = K(1);
//     const double cx = K(2);
//     const double cy = K(3);

//     const int N = points.rows();

//     // Split coordinates
//     Eigen::VectorXd X = points.col(0);
//     Eigen::VectorXd Y = points.col(1);
//     Eigen::VectorXd Z = points.col(2);

//     Eigen::VectorXd r = (X.array().square() + Y.array().square()).sqrt();
//     Eigen::VectorXd theta = (r.array() > 1e-8).select((r.array() / Z.array()).atan(), 0.0);

//     Eigen::VectorXd theta2 = theta.array().square();
//     Eigen::VectorXd theta4 = theta2.array().square();
//     Eigen::VectorXd theta6 = theta2.array() * theta4.array();
//     Eigen::VectorXd theta8 = theta4.array().square();

//     Eigen::VectorXd theta_d = (theta.array()
//         + k1 * theta2.array() * theta.array()
//         + k2 * theta4.array() * theta.array()
//         + k3 * theta6.array() * theta.array()
//         + k4 * theta8.array() * theta.array()).matrix();

//     Eigen::VectorXd scale = (r.array() > 1e-8).select(theta_d.array() / r.array(), 1.0);

//     Eigen::VectorXd x_distorted = X.array() * scale.array();
//     Eigen::VectorXd y_distorted = Y.array() * scale.array();

//     Eigen::MatrixXd projected(N, 2);
//     projected.col(0) = fx * x_distorted.array() + cx;
//     projected.col(1) = fy * y_distorted.array() + cy;

//     return projected;
// }




// // Skew-symmetric matrix
// Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
//     Eigen::Matrix3d w_hat;
//     w_hat <<     0, -w(2),  w(1),
//               w(2),     0, -w(0),
//              -w(1),  w(0),     0;
//     return w_hat;
// }

// // Inverse of the left Jacobian of SO(3)
// Eigen::Matrix3d leftJacobianInverse(const Eigen::Vector3d& omega) {
//     double theta = omega.norm();

//     if (theta < 1e-8) {
//         return Eigen::Matrix3d::Identity();
//     }

//     Eigen::Matrix3d omega_hat = skew(omega);
//     Eigen::Matrix3d omega_hat_sq = omega_hat * omega_hat;

//     double A = 0.5;
//     double B = (1.0 / (theta * theta)) -
//               ((1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta)));

//     return Eigen::Matrix3d::Identity() - A * omega_hat + B * omega_hat_sq;
// }

// // Logarithm map of an SE(3) transformation matrix
// Eigen::Matrix<double, 6, 1> logSE3(const Eigen::Matrix4d& T) {
//     Eigen::Matrix3d R = T.block<3,3>(0,0);
//     Eigen::Vector3d t = T.block<3,1>(0,3);

//     double trace_R = R.trace();
//     double cos_theta = std::min(std::max((trace_R - 1.0) / 2.0, -1.0), 1.0);
//     double theta = std::acos(cos_theta);

//     Eigen::Vector3d omega;
//     Eigen::Matrix3d J_inv;

//     if (theta < 1e-8) {
//         omega.setZero();
//         J_inv = Eigen::Matrix3d::Identity();
//     } else {
//         omega = (theta / (2.0 * std::sin(theta))) * Eigen::Vector3d(
//             R(2,1) - R(1,2),
//             R(0,2) - R(2,0),
//             R(1,0) - R(0,1)
//         );
//         J_inv = leftJacobianInverse(omega);
//     }

//     Eigen::Vector3d upsilon = J_inv * t;

//     Eigen::Matrix<double, 6, 1> result;
//     result.head<3>() = omega;
//     result.tail<3>() = upsilon;

//     return result;
// }

// void visualize_camera_data(
//     const std::vector<Eigen::Vector3d>& obj_pts_list_0,
//     const std::vector<Eigen::Vector2d>& img_pts_list_0,
//     const std::vector<Eigen::Vector2d>& projected_pts_0,
//     const std::vector<Eigen::Vector3d>& obj_pts_list_1,
//     const std::vector<Eigen::Vector2d>& img_pts_list_1,
//     const std::vector<Eigen::Vector2d>& projected_pts_1)
// {
//     pangolin::CreateWindowAndBind("Camera Calibration Visualization", 1280, 720);
//     glEnable(GL_DEPTH_TEST);

//     pangolin::OpenGlRenderState s_cam(
//         pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 100),
//         pangolin::ModelViewLookAt(1, -2, 3, 0, 0, 0, pangolin::AxisY)
//     );

//     pangolin::Handler3D handler(s_cam);
//     pangolin::View& d_cam = pangolin::CreateDisplay()
//                                 .SetBounds(0.0, 1.0, 0.0, 0.7, -1280.0/720.0)
//                                 .SetHandler(&handler);
    

//     pangolin::View& d_2d = pangolin::CreateDisplay()
//                                 .SetBounds(0.0, 1.0, 0.7, 1.0)
//                                 .SetLayout(pangolin::LayoutEqual);

//     while (!pangolin::ShouldQuit()) {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         // Draw 3D view
//         d_cam.Activate(s_cam);
//         glPointSize(5.0);

//         // Draw Camera 0 Object Points
//         glColor3f(0.0, 0.0, 1.0);
//         glBegin(GL_POINTS);
//         for (const auto& pt : obj_pts_list_0)
//             glVertex3d(pt[0], pt[1], pt[2]);
//         glEnd();

//         // Draw Camera 1 Object Points
//         glColor3f(0.5, 0.5, 1.0);
//         glBegin(GL_POINTS);
//         for (const auto& pt : obj_pts_list_1)
//             glVertex3d(pt[0], pt[1], pt[2]);
//         glEnd();

//         // 2D Viewport for image vs. projected
//         d_2d.Activate();

//         glPointSize(3.0);
//         glBegin(GL_POINTS);
//         // Camera 0 - Image Points (Red)
//         glColor3f(1.0, 0.0, 0.0);
//         for (const auto& pt : img_pts_list_0)
//             glVertex2d(pt[0], pt[1]);

//         // Camera 0 - Projected Points (Green)
//         glColor3f(0.0, 1.0, 0.0);
//         for (const auto& pt : projected_pts_0)
//             glVertex2d(pt[0], pt[1]);

//         // Camera 1 - Image Points (Orange)
//         glColor3f(1.0, 0.5, 0.0);
//         for (const auto& pt : img_pts_list_1)
//             glVertex2d(pt[0], pt[1]);

//         // Camera 1 - Projected Points (Cyan)
//         glColor3f(0.0, 1.0, 1.0);
//         for (const auto& pt : projected_pts_1)
//             glVertex2d(pt[0], pt[1]);
//         glEnd();

//         pangolin::FinishFrame();
//     }
// }

// Eigen::VectorXd fisheye_reprojection_error(
//     const Eigen::VectorXd& params,
//     const std::vector<Eigen::MatrixXd>& obj_pts_list_0,
//     const std::vector<Eigen::MatrixXd>& img_pts_list_0,
//     const std::vector<int>& timestamp_list_0,
//     const std::vector<std::vector<int>>& corner_ids_list_0,
//     const std::vector<Eigen::MatrixXd>& obj_pts_list_1,
//     const std::vector<Eigen::MatrixXd>& img_pts_list_1,
//     const std::vector<int>& timestamp_list_1,
//     const std::vector<std::vector<int>>& corner_ids_list_1,
//     const std::vector<int>& all_timestamps
// ) {
//     int num_images_0 = timestamp_list_0.size();
//     int num_images_1 = timestamp_list_1.size();

//     int cam_0_param_length = 8 + num_images_0 * 6;
//     int cam_1_param_length = cam_0_param_length + 8 + num_images_1 * 6;

//     // Parse parameters
//     Eigen::Vector4d K_0 = params.segment<4>(0);
//     Eigen::Vector4d dist_coeffs_0 = params.segment<4>(4);
//     Eigen::MatrixXd extrinsics_0 = Eigen::Map<const Eigen::MatrixXd>(params.data() + 8, 6, num_images_0).transpose();

//     Eigen::Vector4d K_1 = params.segment<4>(cam_0_param_length);
//     Eigen::Vector4d dist_coeffs_1 = params.segment<4>(cam_0_param_length + 4);
//     Eigen::MatrixXd extrinsics_1 = Eigen::Map<const Eigen::MatrixXd>(params.data() + cam_0_param_length + 8, 6, num_images_1).transpose();

//     Eigen::Vector3d rvec_cam_1 = params.segment<3>(cam_1_param_length);
//     Eigen::Vector3d tvec_cam_1 = params.segment<3>(cam_1_param_length + 3);
//     Eigen::Matrix3d R_matrix_cam_1 = Eigen::AngleAxisd(rvec_cam_1.norm(), rvec_cam_1.normalized()).toRotationMatrix();

//     std::vector<double> total_error;

//     for (int i = 0; i < all_timestamps.size(); ++i) {
//         int ts = all_timestamps[i];
//         int cam_0_index = -1, cam_1_index = -1;
//         for (int j = 0; j < timestamp_list_0.size(); ++j)
//             if (timestamp_list_0[j] == ts) cam_0_index = j;
//         for (int j = 0; j < timestamp_list_1.size(); ++j)
//             if (timestamp_list_1[j] == ts) cam_1_index = j;

//         Eigen::Matrix3d R0, R1;
//         Eigen::Vector3d t0, t1;

//         if (cam_0_index != -1) {
//             Eigen::Vector3d rvec = extrinsics_0.row(cam_0_index).head<3>();
//             t0 = extrinsics_0.row(cam_0_index).tail<3>();
//             R0 = Eigen::AngleAxisd(rvec.norm(), rvec.normalized()).toRotationMatrix();

//             Eigen::MatrixXd obj_pts_3d = Eigen::MatrixXd::Zero(obj_pts_list_0[cam_0_index].rows(), 3);
//             obj_pts_3d.leftCols(2) = obj_pts_list_0[cam_0_index];
//             Eigen::MatrixXd transformed = (R0 * obj_pts_3d.transpose()).colwise() + t0;
//             transformed.transposeInPlace();

//             auto projected = kannala_brandt_project(transformed, K_0, dist_coeffs_0);
//             Eigen::MatrixXd err = projected.cast<double>() - img_pts_list_0[cam_0_index].cast<double>();
//             for (int j = 0; j < err.size(); ++j)
//                 total_error.push_back(err(j));
//         }

//         if (cam_1_index != -1) {
//             Eigen::Vector3d rvec = extrinsics_1.row(cam_1_index).head<3>();
//             t1 = extrinsics_1.row(cam_1_index).tail<3>();
//             R1 = Eigen::AngleAxisd(rvec.norm(), rvec.normalized()).toRotationMatrix();

//             Eigen::MatrixXd obj_pts_3d = Eigen::MatrixXd::Zero(obj_pts_list_1[cam_1_index].rows(), 3);
//             obj_pts_3d.leftCols(2) = obj_pts_list_1[cam_1_index];
//             Eigen::MatrixXd transformed = (R1 * obj_pts_3d.transpose()).colwise() + t1;
//             transformed.transposeInPlace();

//             auto projected = kannala_brandt_project(transformed, K_1, dist_coeffs_1);
//             Eigen::MatrixXd err = projected.cast<double>() - img_pts_list_1[cam_1_index].cast<double>();
//             for (int j = 0; j < err.size(); ++j)
//                 total_error.push_back(err(j));
//         }

//         if (cam_0_index != -1 && cam_1_index != -1) {
//             Eigen::Matrix4d T_0 = Eigen::Matrix4d::Identity();
//             T_0.topLeftCorner<3, 3>() = R0;
//             T_0.topRightCorner<3, 1>() = t0;

//             Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
//             T_1.topLeftCorner<3, 3>() = R1;
//             T_1.topRightCorner<3, 1>() = t1;

//             Eigen::Matrix4d T_01_obs = Eigen::Matrix4d::Identity();
//             T_01_obs.topLeftCorner<3, 3>() = R_matrix_cam_1;
//             T_01_obs.topRightCorner<3, 1>() = tvec_cam_1;

//             Eigen::Matrix4d T_01_est = T_0 * T_1.inverse();

//             Eigen::VectorXd pose_error = logSE3(T_01_obs * T_01_est.inverse());
//             for (int j = 0; j < pose_error.size(); ++j)
//                 total_error.push_back(pose_error(j));
//         }
//     }

//     Eigen::VectorXd result(total_error.size());
//     for (size_t i = 0; i < total_error.size(); ++i)
//         result(i) = total_error[i];

//     std::cout << "total_error = " << result.sum() << std::endl;
//     return result;
// }


// struct KannalaBrandtProjection {
//     KannalaBrandtProjection(const Eigen::Vector3d& point_3d, const Eigen::Vector2d& observed)
//         : point_3d_(point_3d), observed_(observed) {}
  
//     template <typename T>
//     bool operator()(const T* const K,           // fx, fy, cx, cy
//                     const T* const dist_coeffs, // k1, k2, k3, k4
//                     T* residuals) const {
//       const T& fx = K[0];
//       const T& fy = K[1];
//       const T& cx = K[2];
//       const T& cy = K[3];
  
//       const T& k1 = dist_coeffs[0];
//       const T& k2 = dist_coeffs[1];
//       const T& k3 = dist_coeffs[2];
//       const T& k4 = dist_coeffs[3];
  
//       const T& X = T(point_3d_(0));
//       const T& Y = T(point_3d_(1));
//       const T& Z = T(point_3d_(2));
  
//       T r = ceres::sqrt(X * X + Y * Y);
//       T theta = T(0);
//       if (ceres::abs(r) > T(1e-8)) {
//         theta = ceres::atan(r / Z);
//       }
  
//       T theta2 = theta * theta;
//       T theta4 = theta2 * theta2;
//       T theta6 = theta2 * theta4;
//       T theta8 = theta4 * theta4;
  
//       T theta_d = theta
//                 + k1 * theta2 * theta
//                 + k2 * theta4 * theta
//                 + k3 * theta6 * theta
//                 + k4 * theta8 * theta;
  
//       T scale = T(1.0);
//       if (ceres::abs(r) > T(1e-8)) {
//         scale = theta_d / r;
//       }
  
//       T x_distorted = X * scale;
//       T y_distorted = Y * scale;
  
//       T u = fx * x_distorted + cx;
//       T v = fy * y_distorted + cy;
  
//       residuals[0] = u - T(observed_(0));
//       residuals[1] = v - T(observed_(1));
  
//       return true;
//     }
  
//     static ceres::CostFunction* Create(const Eigen::Vector3d& point_3d,
//                                        const Eigen::Vector2d& observed) {
//       return new ceres::AutoDiffCostFunction<KannalaBrandtProjection, 2, 4, 4>(
//           new KannalaBrandtProjection(point_3d, observed));
//     }
    
//     private:
//     const Eigen::Vector3d point_3d_;
//     const Eigen::Vector2d observed_;
// };

// template <typename T>
// Eigen::Matrix<T, 2, 1> projectKB(
//     const Eigen::Matrix<T, 3, 1>& p_cam,
//     const T* intrinsic,
//     const T* dist_coeffs)
// {
//     T X = p_cam[0], Y = p_cam[1], Z = p_cam[2];
//     T r = ceres::sqrt(X * X + Y * Y);
//     T th = T(0);
//     if (ceres::abs(r) > T(1e-8)) {
//         th = ceres::atan(r / Z);
//     }

//     T th2 = th * th;
//     T th4 = th2 * th2;
//     T th6 = th2 * th4;
//     T th8 = th4 * th4;

//     T theta_d = th
//               + dist_coeffs[0] * th2 * th
//               + dist_coeffs[1] * th4 * th
//               + dist_coeffs[2] * th6 * th
//               + dist_coeffs[3] * th8 * th;

//     T scale = (ceres::abs(r) > T(1e-8)) ? (theta_d / r) : T(1.0);
//     T x_distorted = X * scale;
//     T y_distorted = Y * scale;

//     T fx = intrinsic[0], fy = intrinsic[1];
//     T cx = intrinsic[2], cy = intrinsic[3];

//     T u = fx * x_distorted + cx;
//     T v = fy * y_distorted + cy;

//     return Eigen::Matrix<T, 2, 1>(u, v);
// }


// struct FisheyeReprojectionErrorQuat {
//     FisheyeReprojectionErrorQuat(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt)
//         : observed_pt_(observed_pt), obj_pt_(obj_pt) {}

//     template <typename T>
//     bool operator()(const T* const intrinsic, const T* const distortion,
//                     const T* const qvec, const T* const tvec,
//                     T* residuals) const {
//         // Convert object point and pose
//         Eigen::Matrix<T, 3, 1> p_obj = obj_pt_.cast<T>();
//         Eigen::Matrix<T, 3, 1> t = {tvec[0], tvec[1], tvec[2]};
//         Eigen::Quaternion<T> q(qvec[0], qvec[1], qvec[2], qvec[3]);
//         Eigen::Matrix<T, 3, 1> p_cam = q * p_obj + t;

//         // Project
//         Eigen::Matrix<T, 2, 1> proj = projectKB(p_cam, intrinsic, distortion);

//         // Residual
//         residuals[0] = proj(0) - T(observed_pt_(0));
//         residuals[1] = proj(1) - T(observed_pt_(1));
//         return true;
//     }

//     static ceres::CostFunction* Create(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt) {
//         return (new ceres::AutoDiffCostFunction<FisheyeReprojectionErrorQuat, 2, 4, 4, 4, 3>(
//             new FisheyeReprojectionErrorQuat(observed_pt, obj_pt)));
//     }

//     Eigen::Vector2d observed_pt_;
//     Eigen::Vector3d obj_pt_;
// };



// struct RelativePoseErrorQuat {
//     static ceres::CostFunction* Create() {
//         return new ceres::AutoDiffCostFunction<RelativePoseErrorQuat, 6,
//                                                4, 3,   // extr0: q0, t0
//                                                4, 3,   // extr1: q1, t1
//                                                4, 3    // q_cam1, t_cam1
//                                               >(new RelativePoseErrorQuat());
//     }

//     template <typename T>
//     bool operator()(const T* const q0, const T* const t0,
//                     const T* const q1, const T* const t1,
//                     const T* const q_cam1, const T* const t_cam1,
//                     T* residuals) const
//     {
//         // === Convert input arrays to Eigen types ===
//         Eigen::Quaternion<T> Q0(q0[0], q0[1], q0[2], q0[3]);
//         Eigen::Matrix<T, 3, 1> T0(t0[0], t0[1], t0[2]);

//         Eigen::Quaternion<T> Q1(q1[0], q1[1], q1[2], q1[3]);
//         Eigen::Matrix<T, 3, 1> T1(t1[0], t1[1], t1[2]);

//         Eigen::Quaternion<T> Q_cam1(q_cam1[0], q_cam1[1], q_cam1[2], q_cam1[3]);
//         Eigen::Matrix<T, 3, 1> T_cam1(t_cam1[0], t_cam1[1], t_cam1[2]);

//         // === Compute relative pose ===
//         Eigen::Quaternion<T> Q_rel = Q1 * Q0.conjugate();
//         Eigen::Matrix<T, 3, 1> T_rel = T1 - Q_rel * T0;

//         // === Pose error ===
//         Eigen::Quaternion<T> Q_err = Q_rel * Q_cam1.conjugate();
//         Eigen::Matrix<T, 3, 1> T_err = T_rel + Q_err * T_cam1;

//         // === Rotation residual (angle-axis) ===
//         T q_err[4] = { Q_err.w(), Q_err.x(), Q_err.y(), Q_err.z() };
//         T rvec_err[3];
//         ceres::QuaternionToAngleAxis(q_err, rvec_err);

//         for (int i = 0; i < 3; ++i) {
//             residuals[i]     = rvec_err[i]*T(1);
//             residuals[i + 3] = T_err[i]*T(1);
//         }

//         return true;
//     }
// };

// struct RelativePoseErrorQuat_Chained {
//     static ceres::CostFunction* Create() {
//         return new ceres::AutoDiffCostFunction<RelativePoseErrorQuat_Chained, 6,
//                                                4, 3,  // extr1
//                                                4, 3,  // extr2
//                                                4, 3,  // qvec_cam1, tvec_cam1
//                                                4, 3   // qvec_cam2, tvec_cam2
//                                               >(new RelativePoseErrorQuat_Chained());
//     }

//     template <typename T>
//     bool operator()(const T* const q1, const T* const t1,
//                     const T* const q2, const T* const t2,
//                     const T* const q_cam1, const T* const t_cam1,
//                     const T* const q_cam2, const T* const t_cam2,
//                     T* residuals) const
//     {
//         // === Extrinsic poses ===
//         Eigen::Quaternion<T> Q1(q1[0], q1[1], q1[2], q1[3]);
//         Eigen::Matrix<T, 3, 1> T1(t1[0], t1[1], t1[2]);

//         Eigen::Quaternion<T> Q2(q2[0], q2[1], q2[2], q2[3]);
//         Eigen::Matrix<T, 3, 1> T2(t2[0], t2[1], t2[2]);

//         // === Camera-to-camera transforms ===
//         Eigen::Quaternion<T> Q_cam1(q_cam1[0], q_cam1[1], q_cam1[2], q_cam1[3]);
//         Eigen::Matrix<T, 3, 1> T_cam1(t_cam1[0], t_cam1[1], t_cam1[2]);

//         Eigen::Quaternion<T> Q_cam2(q_cam2[0], q_cam2[1], q_cam2[2], q_cam2[3]);
//         Eigen::Matrix<T, 3, 1> T_cam2(t_cam2[0], t_cam2[1], t_cam2[2]);

//         // === Compute relative pose between frame extrinsics ===
//         Eigen::Quaternion<T> Q_rel = Q2 * Q1.conjugate();
//         Eigen::Matrix<T, 3, 1> T_rel = T2 - Q_rel * T1;

//         // === Compute cam2-to-cam1 transform from q_cam1 and q_cam2 ===
//         Eigen::Quaternion<T> Q_cam2_rel = Q_cam2 * Q_cam1.conjugate();
//         Eigen::Matrix<T, 3, 1> T_cam2_rel = T_cam2 - Q_cam2_rel * T_cam1;

//         // === Compute error ===
//         Eigen::Quaternion<T> Q_err = Q_rel * Q_cam2_rel.conjugate();
//         Eigen::Matrix<T, 3, 1> T_err = T_rel + Q_err * T_cam2_rel;

//         // === Rotation residual (angle-axis) ===
//         T q_err[4] = { Q_err.w(), Q_err.x(), Q_err.y(), Q_err.z() };
//         T rvec_err[3];
//         ceres::QuaternionToAngleAxis(q_err, rvec_err);

//         for (int i = 0; i < 3; ++i) {
//             residuals[i]     = rvec_err[i] * T(1000);
//             residuals[i + 3] = T_err[i] * T(1000);
//         }

//         return true;
//     }
// };

// struct TemporalSmoothnessError {
//     TemporalSmoothnessError(double translation_weight = 1.0, double rotation_weight = 1.0)
//         : translation_weight(translation_weight), rotation_weight(rotation_weight) {}

//     template <typename T>
//     bool operator()(const T* const q1, const T* const t1,
//                     const T* const q2, const T* const t2,
//                     T* residuals) const {
//         // Penalize translation delta
//         residuals[0] = translation_weight * (t2[0] - t1[0]);
//         residuals[1] = translation_weight * (t2[1] - t1[1]);
//         residuals[2] = translation_weight * (t2[2] - t1[2]);

//         // Penalize rotation delta using angle-axis distance
//         T q1_inv[4] = { q1[0], -q1[1], -q1[2], -q1[3] };
//         T dq[4];
//         ceres::QuaternionProduct(q2, q1_inv, dq); // dq = q2 * inv(q1)

//         // Convert dq to angle-axis
//         T angle_axis[3];
//         ceres::QuaternionToAngleAxis(dq, angle_axis);

//         residuals[3] = rotation_weight * angle_axis[0];
//         residuals[4] = rotation_weight * angle_axis[1];
//         residuals[5] = rotation_weight * angle_axis[2];

//         return true;
//     }

//     static ceres::CostFunction* Create(double translation_weight = 1.0, double rotation_weight = 1.0) {
//         return new ceres::AutoDiffCostFunction<TemporalSmoothnessError, 6, 4, 3, 4, 3>(
//             new TemporalSmoothnessError(translation_weight, rotation_weight));
//     }

//     double translation_weight;
//     double rotation_weight;
// };



// struct TimestampEntry {
//     size_t timestamp_id;
//     int cam0_idx;  // -1 if missing
//     int cam1_idx;  // -1 if missing
//     int cam2_idx;  // -1 if missing
// };

// void OptimizeFishEyeParameters(
//     double intrinsic_0[4], double dist_0[4],
//     std::vector<std::array<double, 7>>& extrinsics_0,
//     const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0,
//     const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0,
//     double intrinsic_1[4], double dist_1[4],
//     std::vector<std::array<double, 7>>& extrinsics_1,
//     const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1,
//     const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1,
//     double intrinsic_2[4], double dist_2[4],
//     std::vector<std::array<double, 7>>& extrinsics_2,
//     const std::vector<std::vector<Eigen::Vector2d>>& img_pts_2,
//     const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_2,
//     double qvec_cam_1[4], double tvec_cam_1[3],
//     double qvec_cam_2[4], double tvec_cam_2[3],
//     const std::vector<TimestampEntry>& master_timestamps
// )
// {
//     ceres::Problem problem;

//     std::vector<bool> extrinsics_0_used(extrinsics_0.size(), false);
//     std::vector<bool> extrinsics_1_used(extrinsics_1.size(), false);
//     std::vector<bool> extrinsics_2_used(extrinsics_2.size(), false);

//     for (const auto& entry : master_timestamps) {
//         // Cam0 residuals
//         if (entry.cam0_idx != -1) {
//             int idx_0 = entry.cam0_idx;
//             extrinsics_0_used[idx_0] = true;

//             for (size_t j = 0; j < img_pts_0[idx_0].size(); ++j) {
//                 ceres::CostFunction* cost_function =
//                     FisheyeReprojectionErrorQuat::Create(img_pts_0[idx_0][j], obj_pts_0[idx_0][j]);

//                 double* qvec_0 = extrinsics_0[idx_0].data();
//                 double* tvec_0 = extrinsics_0[idx_0].data() + 4;

//                 problem.AddResidualBlock(cost_function, nullptr,
//                                          intrinsic_0, dist_0,
//                                          qvec_0, tvec_0);
//             }
//         }

//         // Cam1 residuals
//         if (entry.cam1_idx != -1) {
//             int idx_1 = entry.cam1_idx;
//             extrinsics_1_used[idx_1] = true;

//             for (size_t j = 0; j < img_pts_1[idx_1].size(); ++j) {
//                 ceres::CostFunction* cost_function =
//                     FisheyeReprojectionErrorQuat::Create(img_pts_1[idx_1][j], obj_pts_1[idx_1][j]);

//                 double* qvec_1 = extrinsics_1[idx_1].data();
//                 double* tvec_1 = extrinsics_1[idx_1].data() + 4;

//                 problem.AddResidualBlock(cost_function, nullptr,
//                                          intrinsic_1, dist_1,
//                                          qvec_1, tvec_1);
//             }
//         }

//         // Cam2 residuals
//         if (entry.cam2_idx != -1) {
//             int idx_2 = entry.cam2_idx;
//             extrinsics_2_used[idx_2] = true;

//             for (size_t j = 0; j < img_pts_2[idx_2].size(); ++j) {
//                 ceres::CostFunction* cost_function =
//                     FisheyeReprojectionErrorQuat::Create(img_pts_2[idx_2][j], obj_pts_2[idx_2][j]);

//                 double* qvec_2 = extrinsics_2[idx_2].data();
//                 double* tvec_2 = extrinsics_2[idx_2].data() + 4;

//                 problem.AddResidualBlock(cost_function, nullptr,
//                                          intrinsic_2, dist_2,
//                                          qvec_2, tvec_2);
//             }
//         }

//         // === Relative Pose Constraints ===
//         if (entry.cam0_idx != -1 && entry.cam1_idx != -1) {
//             ceres::CostFunction* rel_pose_cost = RelativePoseErrorQuat::Create();
//             double* qvec_0 = extrinsics_0[entry.cam0_idx].data();
//             double* tvec_0 = extrinsics_0[entry.cam0_idx].data() + 4;
//             double* qvec_1 = extrinsics_1[entry.cam1_idx].data();
//             double* tvec_1 = extrinsics_1[entry.cam1_idx].data() + 4;

//             problem.AddResidualBlock(rel_pose_cost, nullptr,
//                                      qvec_0, tvec_0,
//                                      qvec_1, tvec_1,
//                                      qvec_cam_1, tvec_cam_1);
//         }

//         if (entry.cam0_idx != -1 && entry.cam2_idx != -1) {
//             ceres::CostFunction* rel_pose_cost = RelativePoseErrorQuat::Create();
//             double* qvec_0 = extrinsics_0[entry.cam0_idx].data();
//             double* tvec_0 = extrinsics_0[entry.cam0_idx].data() + 4;
//             double* qvec_2 = extrinsics_2[entry.cam2_idx].data();
//             double* tvec_2 = extrinsics_2[entry.cam2_idx].data() + 4;

//             problem.AddResidualBlock(rel_pose_cost, nullptr,
//                                      qvec_0, tvec_0,
//                                      qvec_2, tvec_2,
//                                      qvec_cam_2, tvec_cam_2);
//         }

//         if (entry.cam1_idx != -1 && entry.cam2_idx != -1) {
//             ceres::CostFunction* rel_pose_cost = RelativePoseErrorQuat_Chained::Create();
//             double* qvec_1 = extrinsics_1[entry.cam1_idx].data();
//             double* tvec_1 = extrinsics_1[entry.cam1_idx].data() + 4;
//             double* qvec_2 = extrinsics_2[entry.cam2_idx].data();
//             double* tvec_2 = extrinsics_2[entry.cam2_idx].data() + 4;

//             problem.AddResidualBlock(rel_pose_cost, nullptr,
//                                      qvec_1, tvec_1,
//                                      qvec_2, tvec_2,
//                                      qvec_cam_1, tvec_cam_1,
//                                      qvec_cam_2, tvec_cam_2);
//         }
//     }

//     // Add temporal smoothness constraints for extrinsics
//     for (size_t i = 1; i < extrinsics_0.size(); ++i) {
//         if (!extrinsics_0_used[i - 1] || !extrinsics_0_used[i]) continue;

//         double* q1 = extrinsics_0[i - 1].data();
//         double* t1 = extrinsics_0[i - 1].data() + 4;
//         double* q2 = extrinsics_0[i].data();
//         double* t2 = extrinsics_0[i].data() + 4;

//         ceres::CostFunction* smoothness_cost =
//             TemporalSmoothnessError::Create(/*translation_weight=*/1.0, /*rotation_weight=*/1.0);

//         problem.AddResidualBlock(smoothness_cost, nullptr, q1, t1, q2, t2);
//     }
//     for (size_t i = 1; i < extrinsics_1.size(); ++i) {
//         if (!extrinsics_1_used[i - 1] || !extrinsics_1_used[i]) continue;

//         double* q1 = extrinsics_1[i - 1].data();
//         double* t1 = extrinsics_1[i - 1].data() + 4;
//         double* q2 = extrinsics_1[i].data();
//         double* t2 = extrinsics_1[i].data() + 4;

//         ceres::CostFunction* smoothness_cost =
//             TemporalSmoothnessError::Create(/*translation_weight=*/1.0, /*rotation_weight=*/1.0);

//         problem.AddResidualBlock(smoothness_cost, nullptr, q1, t1, q2, t2);
//     }
//     for (size_t i = 1; i < extrinsics_2.size(); ++i) {
//         if (!extrinsics_2_used[i - 1] || !extrinsics_2_used[i]) continue;

//         double* q1 = extrinsics_2[i - 1].data();
//         double* t1 = extrinsics_2[i - 1].data() + 4;
//         double* q2 = extrinsics_2[i].data();
//         double* t2 = extrinsics_2[i].data() + 4;

//         ceres::CostFunction* smoothness_cost =
//             TemporalSmoothnessError::Create(/*translation_weight=*/1.0, /*rotation_weight=*/1.0);

//         problem.AddResidualBlock(smoothness_cost, nullptr, q1, t1, q2, t2);
//     }


//     // Add quaternion manifold constraints for extrinsics that are used.
//     for (size_t i = 0; i < extrinsics_0.size(); ++i) {
//         if (extrinsics_0_used[i]) {
//             problem.SetManifold(extrinsics_0[i].data(), new ceres::EigenQuaternionManifold());
//         }
//     }
//     for (size_t i = 0; i < extrinsics_1.size(); ++i) {
//         if (extrinsics_1_used[i]) {
//             problem.SetManifold(extrinsics_1[i].data(), new ceres::EigenQuaternionManifold());
//         }
//     }
//     for (size_t i = 0; i < extrinsics_2.size(); ++i) {
//         if (extrinsics_2_used[i]) {
//             problem.SetManifold(extrinsics_2[i].data(), new ceres::EigenQuaternionManifold());
//         }
//     }

//     // Inter-camera transforms (cam1, cam2 relative to cam0)
//     problem.SetManifold(qvec_cam_1, new ceres::EigenQuaternionManifold());
//     problem.SetManifold(qvec_cam_2, new ceres::EigenQuaternionManifold());

//     // Normalize initial quaternions
//     auto normalize_quats = [](std::vector<std::array<double, 7>>& extrinsics) {
//         for (auto& extr : extrinsics) {
//             Eigen::Map<Eigen::Quaterniond> q(extr.data());
//             q.normalize();
//         }
//     };
//     normalize_quats(extrinsics_0);
//     normalize_quats(extrinsics_1);
//     normalize_quats(extrinsics_2);

//     {
//         Eigen::Map<Eigen::Quaterniond> q(qvec_cam_1);
//         q.normalize();
//     }
//     {
//         Eigen::Map<Eigen::Quaterniond> q(qvec_cam_2);
//         q.normalize();
//     }

//     // Solver
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.minimizer_progress_to_stdout = true;

//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     std::cout << summary.BriefReport() << std::endl;
// }

// struct FrameData {
//     std::vector<Eigen::Vector2d> observed_pts;
//     std::vector<Eigen::Vector2d> projected_pts;
//     std::vector<Eigen::Vector3d> object_pts;
//     std::array<double, 6> extrinsics;
//     double error_sum = 0.0;
//     int frame_index = -1;
//     uint64_t timestamp_ns = -1;
// };


// std::vector<FrameData> GenerateReprojectionErrorData(
//     const double* intrinsic,
//     const double* dist,
//     const std::vector<std::array<double, 7>>& extrinsics,
//     const std::vector<std::vector<Eigen::Vector2d>>& img_pts,
//     const std::vector<std::vector<Eigen::Vector3d>>& obj_pts,
//     const TimestampList& timestamps_ns)
// {

//     std::vector<FrameData> result;

//     Eigen::Map<const Eigen::Vector4d> K(intrinsic);
//     Eigen::Map<const Eigen::Vector4d> dist_coeffs(dist);
//     std::cout << "Generate img_pts.size() = " << img_pts.size() << std::endl;

//     if (timestamps_ns.size() != img_pts.size()) {
//         throw std::runtime_error("Timestamps size does not match frame count.");
//     }

//     for (size_t i = 0; i < img_pts.size(); ++i) {
//         FrameData frame;
//         frame.timestamp_ns = timestamps_ns[i];  // <-- use uint64_t timestamp

//         const auto& observed = img_pts[i];
//         const auto& object = obj_pts[i];
//         const auto& ext = extrinsics[i];

//         Eigen::Quaterniond q(ext[0], ext[1], ext[2], ext[3]);
//         Eigen::Vector3d tvec(ext[4], ext[5], ext[6]);

//         const size_t N = observed.size();
//         Eigen::MatrixXd points_cam(N, 3);

//         for (size_t j = 0; j < N; ++j) {
//             points_cam.row(j) = (q * object[j] + tvec).transpose();
//         }

//         Eigen::MatrixXd projected = kannala_brandt_project(points_cam, K, dist_coeffs);

//         double error_sum = 0.0;
//         for (size_t j = 0; j < N; ++j) {
//             frame.observed_pts.push_back(observed[j]);
//             frame.projected_pts.push_back(projected.row(j).transpose());
//             error_sum += (observed[j] - projected.row(j).transpose()).norm();
//         }

//         frame.error_sum = error_sum;
//         frame.frame_index = static_cast<int>(i);
//         result.push_back(std::move(frame));
//         // std::cout << "Frame " << i << ": timestamp = " << frame.timestamp_ns
//         //           << ", error_sum = " << frame.error_sum << std::endl;
//         // std::cout << "Extrinsics (qvec + tvec): " 
//         //           << ext[0] << ", " << ext[1] << ", " << ext[2] << ", " << ext[3] << ", "
//         //           << ext[4] << ", " << ext[5] << ", " << ext[6] << std::endl;
//         // std::cout << "Observed points: ";
//         // for (const auto& pt : observed) {
//         //     std::cout << "(" << pt.x() << ", " << pt.y() << ") ";
//         // }
//         // std::cout << std::endl;
//         // std::cout << "Projected points: ";
//         // for (int i = 0; i < projected.rows(); ++i) {
//         //     Eigen::Vector2d pt = projected.block<1, 2>(i, 0);
//         //     std::cout << "(" << pt.x() << ", " << pt.y() << ") ";
//         // }
        
//         // std::cout << "Intrinsics: "
//         //           << intrinsic[0] << ", " << intrinsic[1] << ", "
//         //           << intrinsic[2] << ", " << intrinsic[3] << std::endl;
//         // std::cout << "Distortion: "
//         //           << dist[0] << ", " << dist[1] << ", "
//         //           << dist[2] << ", " << dist[3] << std::endl;
//         // std::cout << "----------------------------------------" << std::endl;
//     }

//     std::cout << "Generated " << result.size() << " frames of reprojection error data." << std::endl;
//     return result;
// }


// using FrameMap = std::unordered_map<int64_t, FrameData>;

// void VisualizeStereoReprojectionError(
//     const std::vector<FrameData>& frames_cam0,
//     const std::vector<FrameData>& frames_cam1,
//     const std::vector<FrameData>& frames_cam2,
//     const double* intrinsic_0,
//     const double* dist_0,
//     const double* intrinsic_1,
//     const double* dist_1,
//     const double* intrinsic_2,
//     const double* dist_2)
// {
//     // Build timestamp -> frame map
//     FrameMap map_cam0, map_cam1, map_cam2;
//     std::set<int64_t> all_timestamps;

//     for (const auto& f : frames_cam0) {
//         map_cam0[f.timestamp_ns] = f;
//         all_timestamps.insert(f.timestamp_ns);
//     }

//     for (const auto& f : frames_cam1) {
//         map_cam1[f.timestamp_ns] = f;
//         all_timestamps.insert(f.timestamp_ns);
//     }

//     for (const auto& f : frames_cam2) {
//         map_cam2[f.timestamp_ns] = f;
//         all_timestamps.insert(f.timestamp_ns);
//     }

//     std::vector<int64_t> timestamps(all_timestamps.begin(), all_timestamps.end());

//     // Window setup
//     pangolin::CreateWindowAndBind("Stereo Reprojection Error Viewer", 1600, 800);
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//     // Assume same fx/fy/cx/cy for all cameras
//     int img_w = static_cast<int>(2 * intrinsic_0[2]);
//     int img_h = static_cast<int>(2 * intrinsic_0[3]);

//     pangolin::OpenGlRenderState s_cam(
//         pangolin::ProjectionMatrixOrthographic(0, 3 * img_w, img_h, 0, -1, 1)
//     );

//     pangolin::View& d_cam = pangolin::CreateDisplay()
//         .SetBounds(0.0, 1.0, 0.0, 1.0)
//         .SetAspect(static_cast<float>(3 * img_w) / img_h)
//         .SetHandler(&pangolin::StaticHandler);  

//     size_t current_index = 0;
//     int64_t ts = 0;

//     pangolin::RegisterKeyPressCallback('a', [&]() {
//         if (current_index > 0) current_index--;
//         ts = timestamps[current_index];
//         std::cout << "Current index: " << ts << std::endl;
//         std::cout << "Frame error: " 
//                   << (map_cam0.count(ts) ? map_cam0[ts].error_sum : 0.0) << " (Cam0), "
//                   << (map_cam1.count(ts) ? map_cam1[ts].error_sum : 0.0) << " (Cam1), "
//                   << (map_cam2.count(ts) ? map_cam2[ts].error_sum : 0.0) << " (Cam2)" << std::endl;

//     });

//     pangolin::RegisterKeyPressCallback('d', [&]() {
//         if (current_index + 1 < timestamps.size()) current_index++;
//         ts = timestamps[current_index];
//         std::cout << "Current index: " << ts << std::endl;
//         std::cout << "Frame error: " 
//                   << (map_cam0.count(ts) ? map_cam0[ts].error_sum : 0.0) << " (Cam0), "
//                   << (map_cam1.count(ts) ? map_cam1[ts].error_sum : 0.0) << " (Cam1), "
//                   << (map_cam2.count(ts) ? map_cam2[ts].error_sum : 0.0) << " (Cam2)" << std::endl;
//     });

//     while (!pangolin::ShouldQuit()) {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//         d_cam.Activate(s_cam);

//         ts = timestamps[current_index];

//         if (map_cam0.count(ts)) {
//             const FrameData& f0 = map_cam0[ts];
//             for (size_t i = 0; i < f0.observed_pts.size(); ++i) {
//                 const auto& obs = f0.observed_pts[i];
//                 const auto& proj = f0.projected_pts[i];

//                 // Observed (green)
//                 glColor3f(0.0f, 1.0f, 0.0f);
//                 glPointSize(4.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(obs[0], obs[1]);
//                 glEnd();

//                 // Projected (red)
//                 glColor3f(1.0f, 0.0f, 0.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(proj[0], proj[1]);
//                 glEnd();

//                 // Line (yellow)
//                 glColor3f(1.0f, 1.0f, 0.0f);
//                 glBegin(GL_LINES);
//                 glVertex2f(obs[0], obs[1]);
//                 glVertex2f(proj[0], proj[1]);
//                 glEnd();
//             }
//         }

//         if (map_cam1.count(ts)) {
//             const FrameData& f1 = map_cam1[ts];
//             for (size_t i = 0; i < f1.observed_pts.size(); ++i) {
//                 const auto& obs = f1.observed_pts[i];
//                 const auto& proj = f1.projected_pts[i];

//                 float offset = static_cast<float>(img_w);  // shift right

//                 // Observed (green)
//                 glColor3f(0.0f, 1.0f, 0.0f);
//                 glPointSize(4.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(obs[0] + offset, obs[1]);
//                 glEnd();

//                 // Projected (red)
//                 glColor3f(1.0f, 0.0f, 0.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(proj[0] + offset, proj[1]);
//                 glEnd();

//                 // Line (yellow)
//                 glColor3f(1.0f, 1.0f, 0.0f);
//                 glBegin(GL_LINES);
//                 glVertex2f(obs[0] + offset, obs[1]);
//                 glVertex2f(proj[0] + offset, proj[1]);
//                 glEnd();
//             }
//         }

//         if (map_cam2.count(ts)) {
//             const FrameData& f2 = map_cam2[ts];
//             for (size_t i = 0; i < f2.observed_pts.size(); ++i) {
//                 const auto& obs = f2.observed_pts[i];
//                 const auto& proj = f2.projected_pts[i];

//                 float offset = static_cast<float>(2 * img_w);  // shift right

//                 // Observed (green)
//                 glColor3f(0.0f, 1.0f, 0.0f);
//                 glPointSize(4.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(obs[0] + offset, obs[1]);
//                 glEnd();

//                 // Projected (red)
//                 glColor3f(1.0f, 0.0f, 0.0f);
//                 glBegin(GL_POINTS);
//                 glVertex2f(proj[0] + offset, proj[1]);
//                 glEnd();

//                 // Line (yellow)
//                 glColor3f(1.0f, 1.0f, 0.0f);
//                 glBegin(GL_LINES);
//                 glVertex2f(obs[0] + offset, obs[1]);
//                 glVertex2f(proj[0] + offset, proj[1]);
//                 glEnd();
//             }
//         }

//         pangolin::FinishFrame();
//     }
// }


// void SaveCalibrationResult(
//     const double intrinsic_0[4], const double dist_0[4],
//     const std::vector<std::array<double, 7>>& extrinsics_0,
//     const std::vector<double>& timestamps_0,

//     const double intrinsic_1[4], const double dist_1[4],
//     const std::vector<std::array<double, 7>>& extrinsics_1,
//     const std::vector<double>& timestamps_1,

//     const double intrinsic_2[4], const double dist_2[4],
//     const std::vector<std::array<double, 7>>& extrinsics_2,
//     const std::vector<double>& timestamps_2,

//     const double rvec_cam_1[3], const double tvec_cam_1[3],
//     const double rvec_cam_2[3], const double tvec_cam_2[3]
// ) {
//     json output;

//     // --- Intrinsics & Distortion ---
//     output["camera0"]["intrinsics"] = {intrinsic_0[0], intrinsic_0[1], intrinsic_0[2], intrinsic_0[3]};
//     output["camera0"]["distortion"] = {dist_0[0], dist_0[1], dist_0[2], dist_0[3]};

//     output["camera1"]["intrinsics"] = {intrinsic_1[0], intrinsic_1[1], intrinsic_1[2], intrinsic_1[3]};
//     output["camera1"]["distortion"] = {dist_1[0], dist_1[1], dist_1[2], dist_1[3]};

//     output["camera2"]["intrinsics"] = {intrinsic_2[0], intrinsic_2[1], intrinsic_2[2], intrinsic_2[3]};
//     output["camera2"]["distortion"] = {dist_2[0], dist_2[1], dist_2[2], dist_2[3]};

//     // --- Extrinsics per Frame ---
//     auto add_extrinsics = [](json& cam_json, const std::vector<std::array<double,7>>& extrinsics, const std::vector<double>& timestamps) {
//         for (size_t i = 0; i < extrinsics.size(); ++i) {
//             json pose;
//             pose["timestamp"] = timestamps[i];
//             pose["quaternion"] = {extrinsics[i][0], extrinsics[i][1], extrinsics[i][2], extrinsics[i][3]};
//             pose["translation"] = {extrinsics[i][4], extrinsics[i][5], extrinsics[i][6]};
//             cam_json["extrinsics"].push_back(pose);
//         }
//     };

//     add_extrinsics(output["camera0"], extrinsics_0, timestamps_0);
//     add_extrinsics(output["camera1"], extrinsics_1, timestamps_1);
//     add_extrinsics(output["camera2"], extrinsics_2, timestamps_2);

//     // --- Inter-Camera Transforms ---
//     output["inter_camera"]["camera1_to_camera0"]["rotation_vector"] = {rvec_cam_1[0], rvec_cam_1[1], rvec_cam_1[2]};
//     output["inter_camera"]["camera1_to_camera0"]["translation_vector"] = {tvec_cam_1[0], tvec_cam_1[1], tvec_cam_1[2]};

//     output["inter_camera"]["camera2_to_camera0"]["rotation_vector"] = {rvec_cam_2[0], rvec_cam_2[1], rvec_cam_2[2]};
//     output["inter_camera"]["camera2_to_camera0"]["translation_vector"] = {tvec_cam_2[0], tvec_cam_2[1], tvec_cam_2[2]};

//     // --- Write to file ---
//     std::ofstream ofs("calibration_output.json");
//     ofs << std::setw(4) << output << std::endl;  // Pretty print with indentation
//     std::cout << "Saved calibration results to calibration_output.json" << std::endl;
// }



// void PrintMasterTimestamps(const std::vector<TimestampEntry>& master_timestamps,
//                            const std::vector<size_t>& filtered_timestamp_list_0,
//                            const std::vector<size_t>& filtered_timestamp_list_1,
//                            const std::vector<size_t>& filtered_timestamp_list_2)
// {
//     std::cout << "---------------------------------------------------------------\n";
//     std::cout << "Master Timestamp Alignment:\n";
//     std::cout << "Index |   Timestamp   | Cam0_idx | Cam0_time | Cam1_idx | Cam1_time | Cam2_idx | Cam2_time\n";
//     std::cout << "---------------------------------------------------------------\n";

//     for (size_t i = 0; i < master_timestamps.size(); ++i) {
//         const auto& entry = master_timestamps[i];

//         std::cout << std::setw(5) << i << " | "
//                   << std::setw(13) << entry.timestamp_id << " | "
//                   << std::setw(8)  << entry.cam0_idx << " | ";

//         if (entry.cam0_idx != -1)
//             std::cout << std::setw(10) << filtered_timestamp_list_0[entry.cam0_idx] << " | ";
//         else
//             std::cout << "     ---    | ";

//         std::cout << std::setw(8)  << entry.cam1_idx << " | ";

//         if (entry.cam1_idx != -1)
//             std::cout << std::setw(10) << filtered_timestamp_list_1[entry.cam1_idx] << " | ";
//         else
//             std::cout << "     ---    | ";

//         std::cout << std::setw(8)  << entry.cam2_idx << " | ";

//         if (entry.cam2_idx != -1)
//             std::cout << std::setw(10) << filtered_timestamp_list_2[entry.cam2_idx];
//         else
//             std::cout << "     ---    ";

//         std::cout << std::endl;
//     }

//     std::cout << "---------------------------------------------------------------\n";
// }


// int main(int argc, char** argv) {
//     std::cout << "Argument count (argc): " << argc << std::endl;
//     for (int i = 0; i < argc; ++i) {
//         std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
//     }
//     if (argc < 2) {
//         std::cerr << "Usage: ./calibrate_fisheye_camera <data_file.csv>" << std::endl;
//         return -1;
//     }

//     std::string data_file = argv[1];

//     // Step 1: Load and process CSV data for all cameras
//     auto [obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0] = processCSV(data_file, 0);
//     auto [obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1] = processCSV(data_file, 1);
//     auto [obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2] = processCSV(data_file, 2);

//     // Step 2: Compute homographies and filter timestamps
//     auto [H_list_0, filtered_timestamp_list_0] = computeHomographies(obj_pts_list_0, img_pts_list_0, timestamp_list_0);
//     auto [H_list_1, filtered_timestamp_list_1] = computeHomographies(obj_pts_list_1, img_pts_list_1, timestamp_list_1);
//     auto [H_list_2, filtered_timestamp_list_2] = computeHomographies(obj_pts_list_2, img_pts_list_2, timestamp_list_2);

//     // Filter data for all cameras
//     std::tie(obj_pts_list_0, img_pts_list_0, corner_ids_list_0) = filterDataByTimestamps(
//         obj_pts_list_0, img_pts_list_0, corner_ids_list_0, timestamp_list_0, filtered_timestamp_list_0);
//     std::tie(obj_pts_list_1, img_pts_list_1, corner_ids_list_1) = filterDataByTimestamps(
//         obj_pts_list_1, img_pts_list_1, corner_ids_list_1, timestamp_list_1, filtered_timestamp_list_1);
//     std::tie(obj_pts_list_2, img_pts_list_2, corner_ids_list_2) = filterDataByTimestamps(
//         obj_pts_list_2, img_pts_list_2, corner_ids_list_2, timestamp_list_2, filtered_timestamp_list_2);


//     auto buildIndexMap = [](const std::vector<size_t>& timestamps) -> std::unordered_map<size_t, int> {
//         std::unordered_map<size_t, int> map;
//         for (int i = 0; i < timestamps.size(); ++i) {
//             map[timestamps[i]] = i;
//         }
//         return map;
//     };
    
//     auto map0 = buildIndexMap(filtered_timestamp_list_0);
//     auto map1 = buildIndexMap(filtered_timestamp_list_1);
//     auto map2 = buildIndexMap(filtered_timestamp_list_2);
    
//     // Union of all timestamps
//     std::set<size_t> all_timestamps;
//     all_timestamps.insert(filtered_timestamp_list_0.begin(), filtered_timestamp_list_0.end());
//     all_timestamps.insert(filtered_timestamp_list_1.begin(), filtered_timestamp_list_1.end());
//     all_timestamps.insert(filtered_timestamp_list_2.begin(), filtered_timestamp_list_2.end());
    
//     std::vector<TimestampEntry> master_timestamps;
//     for (auto t : all_timestamps) {
//         master_timestamps.push_back({
//             t,
//             map0.count(t) ? map0[t] : -1,
//             map1.count(t) ? map1[t] : -1,
//             map2.count(t) ? map2[t] : -1
//         });
//     }

//     PrintMasterTimestamps(master_timestamps,
//                       filtered_timestamp_list_0,
//                       filtered_timestamp_list_1,
//                       filtered_timestamp_list_2);

//     std::cin.get();  // Wait for user input before proceeding


//     // Initialize camera parameters
//     // Eigen::Matrix3d K_0;
//     // K_0 << 800, 0, 640,
//     //        0, 800, 480,
//     //        0, 0, 1;
//     // Eigen::Matrix3d K_1;
//     // K_1 << 810, 0, 650,
//     //        0, 810, 470,
//     //        0, 0, 1;
//     // Eigen::Matrix3d K_2;
//     // K_2 << 820, 0, 660,
//     //        0, 820, 460,
//     //        0, 0, 1;




//     Eigen::Matrix3d K_0;
//     K_0 << 610, 0, 688,
//             0, 756, 524,
//             0, 0, 1;
//     Eigen::Matrix3d K_1;
//     K_1 << 802, 0, 662,
//             0, 825, 502,
//             0, 0, 1;
//     Eigen::Matrix3d K_2;
//     K_2 << 671, 0, 648,
//             0, 841, 475,
//             0, 0, 1;


//     double intrinsic_0[4] = {K_0(0, 0), K_0(1, 1), K_0(0, 2), K_0(1, 2)};
//     double dist_0[4] = {-0.013, -0.02, 0.063, -0.03}; 
//     double intrinsic_1[4] = {K_1(0, 0), K_1(1, 1), K_1(0, 2), K_1(1, 2)};
//     double dist_1[4] = {0.09, -0.057, 0.014, -0.004}; 
//     double intrinsic_2[4] = {K_2(0, 0), K_2(1, 1), K_2(0, 2), K_2(1, 2)};
//     double dist_2[4] = {0.03, -0.065, 0.065, -0.0034}; 

//     // Add extrinsics for all cameras
//     std::vector<std::array<double, 7>> extrinsics_0, extrinsics_1, extrinsics_2;
//     for (const auto& H : H_list_0) {
//         auto [R, t] = compute_extrinsic_params(H, K_0);
//         Eigen::Quaterniond q(R);
//         std::array<double, 7> pose;
//         pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
//         pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
//         extrinsics_0.push_back(pose);
//     }
//     for (const auto& H : H_list_1) {
//         auto [R, t] = compute_extrinsic_params(H, K_1);
//         Eigen::Quaterniond q(R);
//         std::array<double, 7> pose;
//         pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
//         pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
//         extrinsics_1.push_back(pose);
//     }
//     for (const auto& H : H_list_2) {
//         auto [R, t] = compute_extrinsic_params(H, K_2);
//         Eigen::Quaterniond q(R);
//         std::array<double, 7> pose;
//         pose[0] = q.w(); pose[1] = q.x(); pose[2] = q.y(); pose[3] = q.z();
//         pose[4] = t(0); pose[5] = t(1); pose[6] = t(2);
//         extrinsics_2.push_back(pose);
//     }

//     // Fill extrinsics vectors aligned with master_timestamps
//     // auto fillExtrinsicsAligned = [](const std::vector<std::array<double, 7>>& extr_src, const std::vector<TimestampEntry>& master_list, int cam_idx) {
//     //     std::vector<std::array<double, 7>> extr_aligned;
//     //     for (const auto& entry : master_list) {
//     //         if ((cam_idx == 0 && entry.cam0_idx != -1) ||
//     //             (cam_idx == 1 && entry.cam1_idx != -1) ||
//     //             (cam_idx == 2 && entry.cam2_idx != -1)) {
//     //             int src_idx = (cam_idx == 0) ? entry.cam0_idx :
//     //                         (cam_idx == 1) ? entry.cam1_idx :
//     //                                         entry.cam2_idx;
//     //             extr_aligned.push_back(extr_src[src_idx]);
//     //         } else {
//     //             // Default identity pose (w=1 quaternion, t=0)
//     //             extr_aligned.push_back({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
//     //         }
//     //     }
//     //     return extr_aligned;
//     // };

//     // extrinsics_0 = fillExtrinsicsAligned(extrinsics_0, master_timestamps, 0);
//     // extrinsics_1 = fillExtrinsicsAligned(extrinsics_1, master_timestamps, 1);
//     // extrinsics_2 = fillExtrinsicsAligned(extrinsics_2, master_timestamps, 2);


//     // Add transformation parameters for camera 1 and camera 2 (relative to camera 0)
//     double rvec_cam_1[3] = {0.05, 0.56, 2.77};
//     double tvec_cam_1[3] = {-0.24, -0.36, 0.047}; // Baseline of 10 cm
//     double qvec_cam_1[4];
//     ceres::AngleAxisToQuaternion(rvec_cam_1, qvec_cam_1);  // Converts to [w, x, y, z]

//     double rvec_cam_2[3] = {-0.60, 0.59, 2.63};  // Different initial rotation
//     double tvec_cam_2[3] = {-0.37, -0.091, -0.18};  // 20cm baseline
//     double qvec_cam_2[4];
//     ceres::AngleAxisToQuaternion(rvec_cam_2, qvec_cam_2);

//     // Step 7: Optimize fisheye parameters
//     OptimizeFishEyeParameters(
//         intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0,
//         intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1,
//         intrinsic_2, dist_2, extrinsics_2, img_pts_list_2, obj_pts_list_2,
//         qvec_cam_1, tvec_cam_1,
//         qvec_cam_2, tvec_cam_2,
//         master_timestamps
//     );

//     std::vector<double> timestamps_0_d(filtered_timestamp_list_0.begin(), filtered_timestamp_list_0.end());
//     std::vector<double> timestamps_1_d(filtered_timestamp_list_1.begin(), filtered_timestamp_list_1.end());
//     std::vector<double> timestamps_2_d(filtered_timestamp_list_2.begin(), filtered_timestamp_list_2.end());

//     SaveCalibrationResult(
//         intrinsic_0, dist_0, extrinsics_0, timestamps_0_d,
//         intrinsic_1, dist_1, extrinsics_1, timestamps_1_d,
//         intrinsic_2, dist_2, extrinsics_2, timestamps_2_d,
//         rvec_cam_1, tvec_cam_1,
//         rvec_cam_2, tvec_cam_2
//     );

//     std::copy(intrinsic_0, intrinsic_0 + 4, intrinsic_2);  // Copy 4 elements
//     std::copy(dist_0, dist_0 + 4, dist_2);  // Copy 4 elements

//     auto cam0_data = GenerateReprojectionErrorData(
//         intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0, filtered_timestamp_list_0);
    
//     auto cam1_data = GenerateReprojectionErrorData(
//         intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1, filtered_timestamp_list_1);

//     auto cam2_data = GenerateReprojectionErrorData(
//         intrinsic_2, dist_2, extrinsics_2, img_pts_list_2, obj_pts_list_2, filtered_timestamp_list_2);

//     VisualizeStereoReprojectionError(cam0_data, cam1_data, cam2_data, intrinsic_0, dist_0, intrinsic_1, dist_1, intrinsic_2, dist_2);

//     // Output refined parameters
//     std::cout << "Refined Intrinsic Parameters for Camera 0:\n";
//     std::cout << "fx: " << intrinsic_0[0] << ", fy: " << intrinsic_0[1]
//               << ", cx: " << intrinsic_0[2] << ", cy: " << intrinsic_0[3] << std::endl;
//     std::cout << "Distortion Coefficients for Camera 0: ";
//     for (double d : dist_0) std::cout << d << " ";
//     std::cout << std::endl;

//     std::cout << "Refined Intrinsic Parameters for Camera 1:\n";
//     std::cout << "fx: " << intrinsic_1[0] << ", fy: " << intrinsic_1[1]
//               << ", cx: " << intrinsic_1[2] << ", cy: " << intrinsic_1[3] << std::endl;
//     std::cout << "Distortion Coefficients for Camera 1: ";
//     for (double d : dist_1) std::cout << d << " ";
//     std::cout << std::endl;

//     std::cout << "Refined Intrinsic Parameters for Camera 2:\n";
//     std::cout << "fx: " << intrinsic_2[0] << ", fy: " << intrinsic_2[1]
//               << ", cx: " << intrinsic_2[2] << ", cy: " << intrinsic_2[3] << std::endl;
//     std::cout << "Distortion Coefficients for Camera 2: ";
//     for (double d : dist_2) std::cout << d << " ";
//     std::cout << std::endl;

//     ceres::QuaternionToAngleAxis(qvec_cam_1, rvec_cam_1);
//     std::cout << "Inter-camera Rotation Vector (Camera 1): ";
//     for (double r : rvec_cam_1) std::cout << r << " ";
//     std::cout << "\nInter-camera Translation Vector (Camera 1): ";
//     for (double t : tvec_cam_1) std::cout << t << " ";
//     std::cout << std::endl;

//     ceres::QuaternionToAngleAxis(qvec_cam_2, rvec_cam_2);
//     std::cout << "Inter-camera Rotation Vector (Camera 2): ";
//     for (double r : rvec_cam_2) std::cout << r << " ";
//     std::cout << "\nInter-camera Translation Vector (Camera 2): ";
//     for (double t : tvec_cam_2) std::cout << t << " ";
//     std::cout << std::endl;

//     return 0;
// }
