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

    std::vector<Point3dVec> obj_pts_list;
    std::vector<Point2dVec> img_pts_list;
    std::vector<IDVec> corner_ids_list;
    TimestampList timestamp_list;

    for (const auto& [timestamp, data] : grouped_data) {
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


struct KannalaBrandtProjection {
    KannalaBrandtProjection(const Eigen::Vector3d& point_3d, const Eigen::Vector2d& observed)
        : point_3d_(point_3d), observed_(observed) {}
  
    template <typename T>
    bool operator()(const T* const K,           // fx, fy, cx, cy
                    const T* const dist_coeffs, // k1, k2, k3, k4
                    T* residuals) const {
      const T& fx = K[0];
      const T& fy = K[1];
      const T& cx = K[2];
      const T& cy = K[3];
  
      const T& k1 = dist_coeffs[0];
      const T& k2 = dist_coeffs[1];
      const T& k3 = dist_coeffs[2];
      const T& k4 = dist_coeffs[3];
  
      const T& X = T(point_3d_(0));
      const T& Y = T(point_3d_(1));
      const T& Z = T(point_3d_(2));
  
      T r = ceres::sqrt(X * X + Y * Y);
      T theta = T(0);
      if (ceres::abs(r) > T(1e-8)) {
        theta = ceres::atan(r / Z);
      }
  
      T theta2 = theta * theta;
      T theta4 = theta2 * theta2;
      T theta6 = theta2 * theta4;
      T theta8 = theta4 * theta4;
  
      T theta_d = theta
                + k1 * theta2 * theta
                + k2 * theta4 * theta
                + k3 * theta6 * theta
                + k4 * theta8 * theta;
  
      T scale = T(1.0);
      if (ceres::abs(r) > T(1e-8)) {
        scale = theta_d / r;
      }
  
      T x_distorted = X * scale;
      T y_distorted = Y * scale;
  
      T u = fx * x_distorted + cx;
      T v = fy * y_distorted + cy;
  
      residuals[0] = u - T(observed_(0));
      residuals[1] = v - T(observed_(1));
  
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

template <typename T>
Eigen::Matrix<T, 2, 1> projectKB(
    const Eigen::Matrix<T, 3, 1>& p_cam,
    const T* intrinsic,
    const T* dist_coeffs)
{
    T X = p_cam[0], Y = p_cam[1], Z = p_cam[2];
    T r = ceres::sqrt(X * X + Y * Y);
    T th = T(0);
    if (ceres::abs(r) > T(1e-8)) {
        th = ceres::atan(r / Z);
    }

    T th2 = th * th;
    T th4 = th2 * th2;
    T th6 = th2 * th4;
    T th8 = th4 * th4;

    T theta_d = th
              + dist_coeffs[0] * th2 * th
              + dist_coeffs[1] * th4 * th
              + dist_coeffs[2] * th6 * th
              + dist_coeffs[3] * th8 * th;

    T scale = (ceres::abs(r) > T(1e-8)) ? (theta_d / r) : T(1.0);
    T x_distorted = X * scale;
    T y_distorted = Y * scale;

    T fx = intrinsic[0], fy = intrinsic[1];
    T cx = intrinsic[2], cy = intrinsic[3];

    T u = fx * x_distorted + cx;
    T v = fy * y_distorted + cy;

    return Eigen::Matrix<T, 2, 1>(u, v);
}


struct FisheyeReprojectionErrorQuat {
    FisheyeReprojectionErrorQuat(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt)
        : observed_pt_(observed_pt), obj_pt_(obj_pt) {}

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const distortion,
                    const T* const qvec, const T* const tvec,
                    T* residuals) const {
        // Convert object point and pose
        Eigen::Matrix<T, 3, 1> p_obj = obj_pt_.cast<T>();
        Eigen::Matrix<T, 3, 1> t = {tvec[0], tvec[1], tvec[2]};
        Eigen::Quaternion<T> q(qvec[0], qvec[1], qvec[2], qvec[3]);
        Eigen::Matrix<T, 3, 1> p_cam = q * p_obj + t;

        // Project
        Eigen::Matrix<T, 2, 1> proj = projectKB(p_cam, intrinsic, distortion);

        // Residual
        residuals[0] = proj(0) - T(observed_pt_(0));
        residuals[1] = proj(1) - T(observed_pt_(1));
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& observed_pt, const Eigen::Vector3d& obj_pt) {
        return (new ceres::AutoDiffCostFunction<FisheyeReprojectionErrorQuat, 2, 4, 4, 4, 3>(
            new FisheyeReprojectionErrorQuat(observed_pt, obj_pt)));
    }

    Eigen::Vector2d observed_pt_;
    Eigen::Vector3d obj_pt_;
};



struct RelativePoseErrorQuat {
    static ceres::CostFunction* Create() {
        return new ceres::AutoDiffCostFunction<RelativePoseErrorQuat, 6,
                                               4, 3,   // extr0: q0, t0
                                               4, 3,   // extr1: q1, t1
                                               4, 3    // q_cam1, t_cam1
                                              >(new RelativePoseErrorQuat());
    }

    template <typename T>
    bool operator()(const T* const q0, const T* const t0,
                    const T* const q1, const T* const t1,
                    const T* const q_cam1, const T* const t_cam1,
                    T* residuals) const
    {
        // === Convert input arrays to Eigen types ===
        Eigen::Quaternion<T> Q0(q0[0], q0[1], q0[2], q0[3]);
        Eigen::Matrix<T, 3, 1> T0(t0[0], t0[1], t0[2]);

        Eigen::Quaternion<T> Q1(q1[0], q1[1], q1[2], q1[3]);
        Eigen::Matrix<T, 3, 1> T1(t1[0], t1[1], t1[2]);

        Eigen::Quaternion<T> Q_cam1(q_cam1[0], q_cam1[1], q_cam1[2], q_cam1[3]);
        Eigen::Matrix<T, 3, 1> T_cam1(t_cam1[0], t_cam1[1], t_cam1[2]);

        // === Compute relative pose ===
        Eigen::Quaternion<T> Q_rel = Q1 * Q0.conjugate();
        Eigen::Matrix<T, 3, 1> T_rel = T1 - Q_rel * T0;

        // === Pose error ===
        Eigen::Quaternion<T> Q_err = Q_rel * Q_cam1.conjugate();
        Eigen::Matrix<T, 3, 1> T_err = T_rel + Q_err * T_cam1;

        // === Rotation residual (angle-axis) ===
        T q_err[4] = { Q_err.w(), Q_err.x(), Q_err.y(), Q_err.z() };
        T rvec_err[3];
        ceres::QuaternionToAngleAxis(q_err, rvec_err);

        for (int i = 0; i < 3; ++i) {
            residuals[i]     = rvec_err[i]*T(1000);
            residuals[i + 3] = T_err[i]*T(1000);
        }

        return true;
    }
};




void OptimizeFishEyeParameters(
    double intrinsic_0[4], double dist_0[4],
    std::vector<std::array<double, 7>>& extrinsics_0,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_0,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_0,
    double intrinsic_1[4], double dist_1[4],
    std::vector<std::array<double, 7>>& extrinsics_1,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_1,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_1,
    // double intrinsic_2[4], double dist_2[4],
    std::vector<std::array<double, 7>>& extrinsics_2,
    const std::vector<std::vector<Eigen::Vector2d>>& img_pts_2,
    const std::vector<std::vector<Eigen::Vector3d>>& obj_pts_2,
    double qvec_cam_1[4], double tvec_cam_1[3],
    double qvec_cam_2[4], double tvec_cam_2[3])
{
    ceres::Problem problem;
    std::vector<int> residuals_cam0(extrinsics_0.size(), 0);
    std::vector<int> residuals_cam1(extrinsics_1.size(), 0);
    std::vector<int> residuals_cam2(extrinsics_2.size(), 0);

    // Camera 0 residuals
    for (size_t i = 0; i < img_pts_0.size(); ++i) {
        residuals_cam0[i] += static_cast<int>(img_pts_0[i].size());
        for (size_t j = 0; j < img_pts_0[i].size(); ++j) {
            ceres::CostFunction* cost_function =
            FisheyeReprojectionErrorQuat::Create(img_pts_0[i][j], obj_pts_0[i][j]);

            double* qvec_0 = extrinsics_0[i].data();       // First 4 elements
            double* tvec_0 = extrinsics_0[i].data() + 4;   // Next 3 elements
            problem.AddResidualBlock(cost_function, nullptr,
                                    intrinsic_0, dist_0,
                                    qvec_0, tvec_0);
            // problem.SetParameterBlockConstant(intrinsic_0);
            // problem.SetParameterBlockConstant(dist_0);
        }
    }

    // Camera 1 residuals
    for (size_t i = 0; i < img_pts_1.size(); ++i) {
        residuals_cam1[i] += static_cast<int>(img_pts_1[i].size());
        for (size_t j = 0; j < img_pts_1[i].size(); ++j) {
            ceres::CostFunction* cost_function =
            FisheyeReprojectionErrorQuat::Create(img_pts_1[i][j], obj_pts_1[i][j]);
            double* qvec_1 = extrinsics_1[i].data();       // First 4 elements
            double* tvec_1 = extrinsics_1[i].data() + 4;   // Next 3 elements
            problem.AddResidualBlock(cost_function, nullptr,
                                    intrinsic_1, dist_1,
                                    qvec_1, tvec_1);
            // problem.SetParameterBlockConstant(intrinsic_1);
            // problem.SetParameterBlockConstant(dist_1);
        }
    }

    // Camera 2 residuals
    for (size_t i = 0; i < img_pts_2.size(); ++i) {
        residuals_cam2[i] += static_cast<int>(img_pts_2[i].size());
        for (size_t j = 0; j < img_pts_2[i].size(); ++j) {
            ceres::CostFunction* cost_function =
                FisheyeReprojectionErrorQuat::Create(img_pts_2[i][j], obj_pts_2[i][j]);
            double* qvec_2 = extrinsics_2[i].data();
            double* tvec_2 = extrinsics_2[i].data() + 4;
            problem.AddResidualBlock(cost_function, nullptr,
                                   intrinsic_0, dist_0,
                                   qvec_2, tvec_2);
            // problem.SetParameterBlockConstant(intrinsic_2);
        }
    }

    std::vector<bool> rel_pose_0_1(extrinsics_0.size(), false);
    std::vector<bool> rel_pose_0_2(extrinsics_0.size(), false);


    // Add relative pose constraints
    for (size_t i = 0; i < extrinsics_0.size(); ++i) {
        if (i >= extrinsics_1.size()) continue;
        ceres::CostFunction* rel_pose_cost = RelativePoseErrorQuat::Create();
        double* qvec_0_rel = extrinsics_0[i].data();       // 4
        double* tvec_0_rel = extrinsics_0[i].data() + 4;   // 3

        double* qvec_1_rel = extrinsics_1[i].data();       // 4
        double* tvec_1_rel = extrinsics_1[i].data() + 4;   // 3

        problem.AddResidualBlock(rel_pose_cost, nullptr,
                                qvec_0_rel, tvec_0_rel,
                                qvec_1_rel, tvec_1_rel,
                                qvec_cam_1, tvec_cam_1);
        // problem.SetParameterBlockConstant(qvec_cam_1);
        // problem.SetParameterBlockConstant(tvec_cam_1);
        rel_pose_0_1[i] = true;
    }

    for (size_t i = 0; i < extrinsics_0.size(); ++i) {
        if (i >= extrinsics_2.size()) continue;
        ceres::CostFunction* rel_pose_cost = RelativePoseErrorQuat::Create();
        double* qvec_0_rel = extrinsics_0[i].data();       // 4
        double* tvec_0_rel = extrinsics_0[i].data() + 4;   // 3

        double* qvec_2_rel = extrinsics_2[i].data();       // 4
        double* tvec_2_rel = extrinsics_2[i].data() + 4;   // 3

        problem.AddResidualBlock(rel_pose_cost, nullptr,
                                qvec_0_rel, tvec_0_rel,
                                qvec_2_rel, tvec_2_rel,
                                qvec_cam_2, tvec_cam_2);
        // problem.SetParameterBlockConstant(qvec_cam_2);
        // problem.SetParameterBlockConstant(tvec_cam_2);
        rel_pose_0_2[i] = true;
    }


    // Add manifold for camera 0 quaternions
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    for (size_t i = 0; i < extrinsics_0.size(); ++i) {
        problem.SetManifold(extrinsics_0[i].data(), quaternion_manifold);
    }

    // Add manifold for camera 1 quaternions
    for (size_t i = 0; i < extrinsics_1.size(); ++i) {
        problem.SetManifold(extrinsics_1[i].data(), quaternion_manifold);
    }

    // Add manifold for camera 2 quaternions
    for (size_t i = 0; i < extrinsics_2.size(); ++i) {
        problem.SetManifold(extrinsics_2[i].data(), quaternion_manifold);
    }

    // Add manifold for camera-to-camera transform quaternions
    problem.SetManifold(qvec_cam_1, quaternion_manifold);
    problem.SetManifold(qvec_cam_2, quaternion_manifold);

    // Normalize all initial quaternions before starting
    for (auto& extr : extrinsics_0) {
        Eigen::Map<Eigen::Quaterniond> q(extr.data());
        q.normalize();
    }
    for (auto& extr : extrinsics_1) {
        Eigen::Map<Eigen::Quaterniond> q(extr.data());
        q.normalize();
    }
    for (auto& extr : extrinsics_2) {
        Eigen::Map<Eigen::Quaterniond> q(extr.data());
        q.normalize();
    }
    {
        Eigen::Map<Eigen::Quaterniond> q(qvec_cam_1);
        q.normalize();
    }
    {
        Eigen::Map<Eigen::Quaterniond> q(qvec_cam_2);
        q.normalize();
    }

    // Print summary table
    std::cout << "\nIndex | Residuals Cam0 | Residuals Cam1 | Residuals Cam2 | RelPose 0-1 | RelPose 0-2\n";
    std::cout << "----------------------------------------------------------------------\n";

    size_t max_idx = extrinsics_0.size();
    for (size_t i = 0; i < max_idx; ++i) {
        std::cout << std::setw(5) << i << " | "
                  << std::setw(14) << residuals_cam0[i] << " | "
                  << std::setw(14) << (i < residuals_cam1.size() ? residuals_cam1[i] : 0) << " | "
                  << std::setw(14) << (i < residuals_cam2.size() ? residuals_cam2[i] : 0) << " | "
                  << std::setw(10) << (rel_pose_0_1[i] ? "Yes" : "No") << " | "
                  << std::setw(10) << (rel_pose_0_2[i] ? "Yes" : "No") << "\n";
    }

    // Configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

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
    const std::vector<std::array<double, 7>>& extrinsics_0,
    const std::vector<double>& timestamps_0,

    const double intrinsic_1[4], const double dist_1[4],
    const std::vector<std::array<double, 7>>& extrinsics_1,
    const std::vector<double>& timestamps_1,

    const double intrinsic_2[4], const double dist_2[4],
    const std::vector<std::array<double, 7>>& extrinsics_2,
    const std::vector<double>& timestamps_2,

    const double rvec_cam_1[3], const double tvec_cam_1[3],
    const double rvec_cam_2[3], const double tvec_cam_2[3]
) {
    json output;

    // --- Intrinsics & Distortion ---
    output["camera0"]["intrinsics"] = {intrinsic_0[0], intrinsic_0[1], intrinsic_0[2], intrinsic_0[3]};
    output["camera0"]["distortion"] = {dist_0[0], dist_0[1], dist_0[2], dist_0[3]};

    output["camera1"]["intrinsics"] = {intrinsic_1[0], intrinsic_1[1], intrinsic_1[2], intrinsic_1[3]};
    output["camera1"]["distortion"] = {dist_1[0], dist_1[1], dist_1[2], dist_1[3]};

    output["camera2"]["intrinsics"] = {intrinsic_2[0], intrinsic_2[1], intrinsic_2[2], intrinsic_2[3]};
    output["camera2"]["distortion"] = {dist_2[0], dist_2[1], dist_2[2], dist_2[3]};

    // --- Extrinsics per Frame ---
    auto add_extrinsics = [](json& cam_json, const std::vector<std::array<double,7>>& extrinsics, const std::vector<double>& timestamps) {
        for (size_t i = 0; i < extrinsics.size(); ++i) {
            json pose;
            pose["timestamp"] = timestamps[i];
            pose["quaternion"] = {extrinsics[i][0], extrinsics[i][1], extrinsics[i][2], extrinsics[i][3]};
            pose["translation"] = {extrinsics[i][4], extrinsics[i][5], extrinsics[i][6]};
            cam_json["extrinsics"].push_back(pose);
        }
    };

    add_extrinsics(output["camera0"], extrinsics_0, timestamps_0);
    add_extrinsics(output["camera1"], extrinsics_1, timestamps_1);
    add_extrinsics(output["camera2"], extrinsics_2, timestamps_2);

    // --- Inter-Camera Transforms ---
    output["inter_camera"]["camera1_to_camera0"]["rotation_vector"] = {rvec_cam_1[0], rvec_cam_1[1], rvec_cam_1[2]};
    output["inter_camera"]["camera1_to_camera0"]["translation_vector"] = {tvec_cam_1[0], tvec_cam_1[1], tvec_cam_1[2]};

    output["inter_camera"]["camera2_to_camera0"]["rotation_vector"] = {rvec_cam_2[0], rvec_cam_2[1], rvec_cam_2[2]};
    output["inter_camera"]["camera2_to_camera0"]["translation_vector"] = {tvec_cam_2[0], tvec_cam_2[1], tvec_cam_2[2]};

    // --- Write to file ---
    std::ofstream ofs("calibration_output.json");
    ofs << std::setw(4) << output << std::endl;  // Pretty print with indentation
    std::cout << "Saved calibration results to calibration_output.json" << std::endl;
}

struct TimestampEntry {
    size_t timestamp_id;
    int cam0_idx;  // -1 if missing
    int cam1_idx;  // -1 if missing
    int cam2_idx;  // -1 if missing
};



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

    // Step 7: Optimize fisheye parameters
    // OptimizeFishEyeParameters(
    //     intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0,
    //     intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1,
    //     intrinsic_2, dist_2, extrinsics_2, img_pts_list_2, obj_pts_list_2,
    //     qvec_cam_1, tvec_cam_1,
    //     qvec_cam_2, tvec_cam_2
    // );
    OptimizeFishEyeParameters(
        intrinsic_0, dist_0, extrinsics_0, img_pts_list_0, obj_pts_list_0,
        intrinsic_1, dist_1, extrinsics_1, img_pts_list_1, obj_pts_list_1,
        extrinsics_2, img_pts_list_2, obj_pts_list_2,
        qvec_cam_1, tvec_cam_1,
        qvec_cam_2, tvec_cam_2
    );

    // SaveCalibrationResult(
    //     intrinsic_0, dist_0, extrinsics_0, filtered_timestamp_list_0,
    //     intrinsic_1, dist_1, extrinsics_1, filtered_timestamp_list_1,
    //     intrinsic_2, dist_2, extrinsics_2, filtered_timestamp_list_2,
    //     rvec_cam_1, tvec_cam_1,
    //     rvec_cam_2, tvec_cam_2
    // );

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
