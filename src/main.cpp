#include <iostream>
#include <string>
#include <memory>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include "omp.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>



#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>


#include <chrono>
#include "math.h"

#include <Eigen/Dense>


#include <fstream>


template<typename M>
M load_csv (const std::string & path) {
    using namespace Eigen;
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    uint rows_= 0;
    while (std::getline(indata, line)) {

        if (rows>=3){
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }

            ++rows_;
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows_);
}


std::ostream& operator<<(std::ostream& o, std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> p){
    for (auto p_:p){
        o<<"( "<<p_.x<<","<<p_.y<<","<<p_.z<<" )"<<std::endl;
    }
    return o;
}

template<typename TYPE>
std::ostream& operator<<(std::ostream& o, std::vector<TYPE> p){
    for (auto p_:p){
        std::cout<<"( "<<p_<<" )"<<std::endl;
    }
    return o;
}


bool isInBetween(Eigen::VectorXf p1, Eigen::VectorXf p2, Eigen::VectorXf a){
    // check if a is between p1 and p2;

    auto p2p1=p2-p1;
    auto p1p2=p1-p2;
    auto p2a=a-p2;
    auto p1a=a-p1;

    auto th1=acos(p2p1.dot(p1a)/(p2p1.norm() * p2a.norm()));
    auto th2=acos(p1p2.dot(p2a)/(p1p2.norm() * p1a.norm()));
    int decision=abs(th1)>=(3.14/2-0.2) || abs(th2)>=(3.14/2-0.2);



    return decision;

}

bool find_k_nearest_neighbour(pcl::PointXYZ searchPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr theCloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr KNNcloud, int K,
                              pcl::KdTreeFLANN<pcl::PointXYZ> kdtree){

    



    std::vector<int> pointIdx(K);
    std::vector<float> distances(K);


    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points_;

    if (kdtree.nearestKSearch(searchPoint, K, pointIdx, distances)){
        
        int size_found = pointIdx.size();
        
        points_.resize(size_found);

        
        for (int i=0; i<size_found; i++){
        
            points_[i]=(*theCloud)[pointIdx[i]];
        
        
        }

        KNNcloud->points=points_;
        return true;

    }
    else {
        return false;
    }

}

void isInside(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> main_cloud,
              std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> out_cloud,
              std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> check_cloud_1, 
              pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_2){

    auto start_checking=std::chrono::steady_clock::now();

    auto inmat=main_cloud->getMatrixXfMap();
    

    inmat=inmat.array().isFinite().select(inmat, 1000.0f);

    
    auto chkmat_1=check_cloud_1->getMatrixXfMap();
    auto chkmat_2=check_cloud_2->getMatrixXfMap();


    auto ch1=chkmat_1.block(0,0,2,chkmat_1.cols());
    auto ch2=chkmat_2.block(0,0,2,chkmat_2.cols());
    
    
    auto in=inmat.block(0,0,2,inmat.cols());

    Eigen::VectorXd mask(in.cols());
    

    #pragma omp parallel for
    for (int i =0; i<in.cols();i++)
    {   

        
        Eigen::VectorXf var=in.col(i);
        auto out_=ch1.colwise()-var;
        auto out__=ch2.colwise()-var;

        Eigen::VectorXf norm_=out_.colwise().norm();
        Eigen::VectorXf norm__=out__.colwise().norm();


        int min_index1;
        float min_val1 = norm_.minCoeff(&min_index1);

        int min_index2;
        float min_val2=norm__.minCoeff(&min_index2);

        Eigen::VectorXf p1 = ch1.block(0,min_index1, 2, 1);
        Eigen::VectorXf p2 = ch2.block(0,min_index2,2,1);
        Eigen::VectorXf a = var;
        
        //mask(i)=1;
        mask(i)=isInBetween(p1,p2,a);
    
    }

    //std::cout<<"the size of the cloud is: " << main_cloud->size()<<" it decreased to this number: "<< mask.sum()<<std::endl;

    for (int i = 0 ; i< mask.size(); i++){
        if (mask(i)<1)
        {   
            out_cloud->points.push_back(main_cloud->points[i]);

        }

    }
    
    


   
    auto end_checking=std::chrono::steady_clock::now();
 

    std::cout<<"Total time took for excluding the point cloud is: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_checking-start_checking).count()<<std::endl;

}


void make_theclouds(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> main_cloud,
                    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> check_cloud_1, pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_2,
                    const int& num_points){
    auto startMakeClouds=std::chrono::steady_clock::now();

    
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points_(num_points);
    
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points_ch1(314);
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points_ch2(314);


     
    float x,y,z;
    x=y=z=0.0f;

    float x_,y_,z_;
    x_=y_=10.0f;
    z_=0.0f;

    float r1=5.0f;
    float r2=5.0f;
    float th=0.0f;


    for( int i =0; i<314; i++){
        
        th-= 3.14/314;
        
        x=r1*cos(th) + x_;
        y=r1*sin(th) + y_;
        
        pcl::PointXYZ p_={x,y,z_};

        points_ch1[i]=p_; 
    }

    x_=y_=10.0f;
    th=0.0f;
    x=y=z=0.0f;

    for(int i =0;i<314;i++){
        th+= 3.14/314;
        x=r2*cos(th) + x_;
        y=r2*sin(th) + y_;
        pcl::PointXYZ p_={x,y,z_};
        points_ch2[i]=p_;
        
    }
    

    x=y=z=0.0f;

    for (int i=0; i< num_points; ++i){
        x+=1.5;
        y=0.0f;
        for (int j = 0 ; j<num_points; ++j){
            y+=1.5;
            z=0.0f;
            for (int k=0; k<num_points; ++k){
                z+=1.5;
                pcl::PointXYZ p_={x,y,z};
                points_.push_back(p_);           
            }
        }
    }

    main_cloud->reserve(num_points);
    main_cloud->points=points_;

    check_cloud_1->reserve(314);
    check_cloud_1->points=points_ch1;
    
    check_cloud_2->reserve(314);
    check_cloud_2->points=points_ch2;

    auto endMakeClouds=std::chrono::steady_clock::now();


    
    
    std::cout<<"Total time took for making 60K cloud and 200 line cloud is: "<<
        std::chrono::duration_cast<std::chrono::milliseconds>(endMakeClouds-startMakeClouds).count()
        <<std::endl;

}


void test(){
    Eigen::VectorXf p1(2);
    Eigen::VectorXf p2(2);
    Eigen::VectorXf a(2);

    p1<<0,0;
    p2<<0,1;
    a<<0.5,0.5;
    isInBetween(p1,p2,a);
}



class ROIdetection{
    public:
    
    ros::NodeHandle nh;
    ros::Subscriber LidarSub;
    ros::Subscriber LidarTransform;
    ros::Publisher ROIpub;

    ros::Publisher pub1;
    ros::Publisher pub2;
        
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr rightCurb;
    pcl::PointCloud<pcl::PointXYZ>::Ptr leftCurb;


    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_right;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_left;
    
    pcl::PointXYZ searchPoint;


    
    Eigen::Matrix4f lLUTM;


    bool csv_read=false;
    bool csv_error=false;

    ROIdetection(){

        lLUTM=Eigen::Matrix4f::Identity();

        LidarSub=nh.subscribe("/rslidar_points_front", 100, &ROIdetection::LidarCallback, this);
        LidarTransform=nh.subscribe("/transforms/LUTM_T_L",10, &ROIdetection::LidarTransformCallback, this);



        ROIpub=nh.advertise<sensor_msgs::PointCloud2>("/ROI_filtered", 100);


        pub1=nh.advertise<sensor_msgs::PointCloud2>("/debug_right", 100);
        pub2=nh.advertise<sensor_msgs::PointCloud2>("/debug_left", 100);


        in_cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        rightCurb=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        leftCurb=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();






        read_csv(rightCurb, leftCurb);

        kdtree_right.setInputCloud(rightCurb);
        kdtree_left.setInputCloud(leftCurb);


    }
    void LidarCallback(const sensor_msgs::PointCloud2::ConstPtr pc2msg){
        
        pcl::fromROSMsg(*pc2msg, *in_cloud);

        if (csv_read){
    
            pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_map(new pcl::PointCloud<pcl::PointXYZ>());
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());




            pcl::PointCloud<pcl::PointXYZ>::Ptr rightCurb_neighbour(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr leftCurb_neighbour(new pcl::PointCloud<pcl::PointXYZ>());




            pcl::transformPointCloud (*in_cloud, *in_cloud_map, lLUTM);

            find_k_nearest_neighbour(searchPoint, rightCurb,
                              rightCurb_neighbour, 200,
                              kdtree_right);
            
            
            find_k_nearest_neighbour(searchPoint, leftCurb,
                              leftCurb_neighbour, 200,
                              kdtree_left);


            isInside(in_cloud_map, out_cloud, leftCurb_neighbour , rightCurb_neighbour);



            pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

            pcl::PassThrough<pcl::PointXYZ> filter;

            filter.setInputCloud(out_cloud);
            filter.setFilterFieldName("z");
            filter.setNegative(true);

            filter.setFilterLimits(-1.0,0.5);

            filter.filter(*plane);








            debug_knn(rightCurb_neighbour, leftCurb_neighbour);

            sensor_msgs::PointCloud2 pc2msg_out;


            pcl::toROSMsg(*plane, pc2msg_out);

            pc2msg_out.header.frame_id="map";
            pc2msg_out.header.stamp=ros::Time::now();

            ROIpub.publish(pc2msg_out);





        }








        else if (!csv_read & !csv_error) {
            std::cout<<"waiting for the csv file"<<std::endl;
        }
        else{

        }
    
    
    }
    void debug_knn(pcl::PointCloud<pcl::PointXYZ>::Ptr rightCurb, pcl::PointCloud<pcl::PointXYZ>::Ptr leftCurb){
        sensor_msgs::PointCloud2 pc2msg_outR;
        sensor_msgs::PointCloud2 pc2msg_outL;


        pcl::toROSMsg(*rightCurb, pc2msg_outR);
        pcl::toROSMsg(*leftCurb, pc2msg_outL);

        pc2msg_outR.header.frame_id=pc2msg_outL.header.frame_id="map";
        pc2msg_outR.header.stamp=pc2msg_outL.header.stamp=ros::Time::now();


        pub1.publish(pc2msg_outR);
        pub2.publish(pc2msg_outL);



    }
    void read_csv(pcl::PointCloud<pcl::PointXYZ>::Ptr rightCurb, pcl::PointCloud<pcl::PointXYZ>::Ptr leftCurb){
        std::string filepath="/home/dastan/pd-ring-road-2022-12-03.csv";
        
        csv_read=true;
        
        Eigen::MatrixXd A = load_csv<Eigen::MatrixXd>(filepath);

        auto size_=A.rows();
        
        auto leftCurb_mat=A.block(0,11, size_, 2);
        auto rightCurb_mat=A.block(0,13, size_, 2);

        rightCurb->resize(size_);
        leftCurb->resize(size_);

        for (int i = 0 ; i<size_; i++){
            
            (*rightCurb)[i] = {rightCurb_mat(i,0), rightCurb_mat(i,1), 0.0f};
            (*leftCurb)[i] = {leftCurb_mat(i,0), leftCurb_mat(i,1), 0.0f};

        
        }
    

    }

    void LidarTransformCallback(std_msgs::Float32MultiArray llutm){
        
        std::vector<float> vec= llutm.data;
    
        auto lllutm=Eigen::Map<Eigen::Matrix4f>(vec.data(), 4,4);
        lLUTM=lllutm.transpose();

        searchPoint.x=lLUTM(0,3);
        searchPoint.y=lLUTM(1,3);
        searchPoint.z=0.0f;
    }

};

int main(int argc, char** argv){
    ros::init(argc,argv,"node");
    int num_points_;


    omp_set_num_threads(12); 
    
    
    if (argc>1){
        num_points_=std::stoi(argv[1]);
    }
    else{
        num_points_=40;
    }


    const int num_points=const_cast<const int&>(num_points_);
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> in_cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> out_cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_2(new pcl::PointCloud<pcl::PointXYZ>());
    
//    make_theclouds(in_cloud, check_cloud_1, check_cloud_2, num_points);

    
    
    bool take_test=true;
   
   
   
    if (take_test){
        
        ROIdetection roi;
        ros::spin();

//        pcl::PointXYZ point={30.0,30.0,0.0};
//        
//        pcl::PointCloud<pcl::PointXYZ>::Ptr knn_cloud(new pcl::PointCloud<pcl::PointXYZ>());
//
//        if ( find_k_nearest_neighbour(point, in_cloud, knn_cloud, 20) ){
//            pcl::visualization::PCLVisualizer viewer("view"); 
//            viewer.addPointCloud(knn_cloud,"a");
//            viewer.spin();
//        } 
    }

    
    else{
    
        const int num_points=const_cast<const int&>(num_points_);

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> in_cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> out_cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_1(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud_2(new pcl::PointCloud<pcl::PointXYZ>());




        
        make_theclouds(in_cloud, check_cloud_1, check_cloud_2, num_points);
        
        isInside(in_cloud, out_cloud, check_cloud_1, check_cloud_2);

        
        
        pcl::visualization::PCLVisualizer viewer("view");        

        viewer.addPointCloud(check_cloud_1, "c");
        viewer.addPointCloud(check_cloud_2, "a");
        //viewer.addPointCloud(in_cloud,"d");
        //viewer.addPolygon(pc,"a");
        viewer.addPointCloud(out_cloud,"b");
        viewer.spin();

    }
  
    return 0;

}

