#include <iostream>
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include <chrono>

void parser(int argc, char** argv, std::string& PCtopic){
    char forwardSlash='/';

    if (argc==1){
        PCtopic="/rslidar_points_front";
    }
    else if (argc > 2){
        printf(".........only one topic at a time...........\n");
    }
    else{
        PCtopic=argv[1];        
    }


    if (PCtopic[0] != '/'){
        PCtopic.insert(0,1,forwardSlash);
    }

}

void PCcallback( const sensor_msgs::PointCloud2::ConstPtr& pcmsg){
    pcl::PointCloud<pcl::PointXYZ> pclCloud;
    pcl::fromROSMsg(*pcmsg, pclCloud);

    std::cout<<"The cloud size is ...."<< pclCloud.size()<<std::endl;


    pcl::PointIndices::Ptr groundIndices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr Coeffsground(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> groundSegmentor;
    
    groundSegmentor.setOptimizeCoefficients(true);
    groundSegmentor.setModelType(pcl::SACMODEL_PLANE);
    groundSegmentor.setMethodType(pcl::SAC_RANSAC);
    groundSegmentor.setMaxIterations(100);
    groundSegmentor.setDistanceThreshold(0.1);
    groundSegmentor.setInputCloud(pclCloud);
    
    auto start = std::chrono::steady_clock::now();
    groundSegmentor.segment(*groundIndices, *Coeffsground);
    auto end = std::chrono::steady_clock::now();
    
    std::cout<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;
    


}

int main(int argc, char** argv){
    std::string PCtopic;

    parser(argc, argv, PCtopic);
    std::cout<<"The topic to subscribe is..."<<PCtopic<<std::endl;

    ros::init(argc, argv,"LidarRadar");
    ros::NodeHandle PCnh;
    ros::Subscriber sub=PCnh.subscribe<sensor_msgs::PointCloud2>(PCtopic, 10, PCcallback);
    

    try{
        ros::spin();
    }
    catch (std::exception& e){
        std::cout<<e.what()<<std::endl;
    }




}

