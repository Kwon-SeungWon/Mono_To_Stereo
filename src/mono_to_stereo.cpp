#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

class StereoDepthNode
{
public:
  StereoDepthNode()
  {
    ros::NodeHandle nh;

    // Subscribers for compressed left and right camera images
    image_transport::ImageTransport it(nh);
    left_cam_sub_.subscribe(nh, "/usb_cam_left/image_raw/compressed", 1);
    right_cam_sub_.subscribe(nh, "/usb_cam_right/image_raw/compressed", 1);

    // Publisher for the disparity image
    disparity_pub_ = it.advertise("/stereo/disparity", 1);

    sync_ = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), left_cam_sub_, right_cam_sub_);
    sync_->registerCallback(boost::bind(&StereoDepthNode::callback, this, _1, _2));

    loadCameraParameters();
    ROS_INFO("StereoDepthNode initialized.");
  }

  ~StereoDepthNode()
  {
    delete sync_;
  }

  void loadCameraParameters()
  {
    camera_intrinsic_matrix = (cv::Mat1d(3, 3) << 462.3399, 0.000000, 355.4217,
                                              0.0000, 461.5362676, 227.7757196,
                                              0.00000, 0.00000, 1.00000);

    camera_distortion_coeffs = (cv::Mat1d(1, 5) << -0.3461702, 
                                               0.1011478, 
                                               0.0054566, 
                                               -0.00063, 
                                               0.0); 
  }

  void callback(const sensor_msgs::CompressedImageConstPtr& left_image_msg, const sensor_msgs::CompressedImageConstPtr& right_image_msg)
  {
    // Decompress compressed images
    cv::Mat left_image = decompressImage(left_image_msg);
    cv::Mat right_image = decompressImage(right_image_msg);

    if (left_image.empty() || right_image.empty())
    {
      ROS_ERROR("Empty image after decompression");
      return;
    }

    // Check if the image sizes are the same
    if (left_image.size() != right_image.size())
    {
      ROS_WARN("Left and right images do not have the same size, resizing right image to match left image.");
      cv::resize(right_image, right_image, left_image.size());
    }

    // Undistort images
    cv::Mat left_image_undistorted, right_image_undistorted;
    cv::undistort(left_image, left_image_undistorted, camera_intrinsic_matrix, camera_distortion_coeffs);
    cv::undistort(right_image, right_image_undistorted, camera_intrinsic_matrix, camera_distortion_coeffs);

    if (left_image_undistorted.size() != right_image_undistorted.size())
    {
      ROS_ERROR("Undistorted left and right images do not have the same size!");
      return;
    }

    // StereoBM parameters
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9);
    cv::Mat disparity, disparity_normalized;

    stereo->compute(left_image_undistorted, right_image_undistorted, disparity);
    cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the results
    cv::imshow("Left Image", left_image_undistorted);
    cv::imshow("Right Image", right_image_undistorted);
    cv::imshow("Disparity", disparity_normalized);
    cv::waitKey(1);

    // Publish the disparity image
    publishDisparity(disparity_normalized);
  }

private:
  cv::Mat decompressImage(const sensor_msgs::CompressedImageConstPtr& compressed_msg)
  {
    try
    {
      // Convert compressed image to OpenCV format
      cv::Mat compressed_image = cv::imdecode(cv::Mat(compressed_msg->data), cv::IMREAD_GRAYSCALE);
      return compressed_image;
    }
    catch (cv::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return cv::Mat();
    }
  }

  void publishDisparity(const cv::Mat& disparity)
  {
    // Convert the disparity OpenCV Mat into a ROS Image message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "stereo_camera_frame";

    sensor_msgs::ImagePtr disparity_msg = cv_bridge::CvImage(header, "mono8", disparity).toImageMsg();
    disparity_pub_.publish(disparity_msg);
  }

  cv::Mat camera_intrinsic_matrix, camera_distortion_coeffs;
  image_transport::Publisher disparity_pub_;
  message_filters::Subscriber<sensor_msgs::CompressedImage> left_cam_sub_, right_cam_sub_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy>* sync_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stereo_depth_node");
  StereoDepthNode stereo_depth_node;
  ros::spin();
  return 0;
}
