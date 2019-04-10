#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <random>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU> 
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;
const double eps = 1e-6;
const double PI = acos(-1);
string origin_Img_Name,bg_Img_Name;
Mat_<cv::Vec3d> origin_Img,bg_Img,temp1,temp2;
cv::Point2d mouse_Position= cv::Point2d(-1, -1),bg_Position;
std::vector<cv::Point2d> my_line;
bool selected_bg;
Mat_<cv::Vec3d> gradient_img;

int dx[] = { 0, 0,-1, 1};
int dy[] = { 1,-1, 0, 0};



//---------------------------------------------------------------------
//---------------------------------------------------------------------
//----------------------------Point in Polygon-------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
int dcmp(double x)
{
    if(fabs(x)<eps) return 0;
    return x<0?-1:1;
}
double Dot(Point2d A,Point2d B)
{
    return A.x*B.x+A.y*B.y;
}
double Cross(Point2d A,Point2d B)
{
    return A.x*B.y-A.y*B.x;
}
bool InSegment(Point2d P,Point2d A,Point2d B)
{
    return dcmp(Cross(A-B,P-A))==0 && dcmp(Dot(A-P,B-P))<=0;
}
bool IsPointInPolygon(Point2d p,vector<Point2d>& poly)
{
    int wn=0;
    int n = poly.size();
    for(int i=0;i<n;++i)
    {
        if(InSegment(p, poly[(i+1)%n], poly[i]) ) return true;
        int k=dcmp( Cross(poly[(i+1)%n]-poly[i], p-poly[i] ) );
        int d1=dcmp( poly[i].y-p.y );
        int d2=dcmp( poly[(i+1)%n].y-p.y );
        if(k>0 && d1<=0 && d2>0) ++wn;
        if(k<0 && d2<=0 && d1>0) --wn;
    }
    if(wn!=0) return true;
    return false;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//----------------------------Point in Polygon-------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------












//---------------------------------------------------------------------
//---------------------------------------------------------------------
//-----------------------Select Mask-----------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
void on_mouse(int event, int x, int y, int flags, void *ustc)
//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号    
{
    Mat_<cv::Vec3d>& image = *(Mat_<cv::Vec3d>*) ustc;
    char temp[16];
    switch (event) {
        case EVENT_LBUTTONDOWN://按下左键
        {
            sprintf(temp, "(%d,%d)", x, y);
            putText(image, temp, cv::Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
            my_line.push_back(cv::Point2d(x,y));
        }break;
        case EVENT_MOUSEMOVE://移动鼠标
        {
            mouse_Position = cv::Point2d(x, y);
        }break;
        // case EVENT_LBUTTONUP:
        // {
        //     sprintf(temp, "(%d,%d)", x, y);
        //     putText(image, temp, cv::Point2d(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        // }break;
    }
    return ;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//-----------------------Select Mask-----------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------













//---------------------------------------------------------------------
//---------------------------------------------------------------------
//-----------------------Select merge point----------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
void on_mouse2(int event, int x, int y, int flags, void *ustc)
{
    Mat_<cv::Vec3d>& image = *(Mat_<cv::Vec3d>*) ustc;
    char temp[16];
    switch (event) {
        case EVENT_LBUTTONDOWN://按下左键
        {
            if (!selected_bg) 
            {
                sprintf(temp, "(%d,%d)", x, y);
                putText(image, temp, cv::Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
                bg_Position = cv::Point2d(x, y);
                selected_bg = 1;
            }
        }break;
        case EVENT_MOUSEMOVE://移动鼠标
        {
            mouse_Position = cv::Point2d(x, y);
        }break;
        // case EVENT_LBUTTONUP:
        // {
        //     sprintf(temp, "(%d,%d)", x, y);
        //     putText(image, temp, cv::Point2d(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        // }break;
    }
    return ;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//-----------------------Select merge point----------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------















//---------------------------------------------------------------------
//---------------------------------------------------------------------
//------------------------------Read image-----------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// 0: have problem
bool read_Img(string& image_Name,Mat_<cv::Vec3d>& readimg)
{
    readimg = imread( image_Name); // Read the file
    if( readimg.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image: " << image_Name << std::endl ;
        return 0;
    }
    return 1;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//------------------------------Read image-----------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------












//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------Calculate gradient of original image------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
void gradient(Mat_<cv::Vec3d>& ori_Img)
{
    int col = ori_Img.cols;
    int row = ori_Img.rows;
    ori_Img.copyTo(gradient_img);
    for (int i=1;i<row-1;i++)
        for (int j=1;j<col-1;j++) 
            gradient_img(i,j) = 4*origin_Img(i,j) - 
                                origin_Img(i-1,j) - 
                                origin_Img(i+1,j) -
                                origin_Img(i,j-1) -
                                origin_Img(i,j+1);
    for (int i=0;i<row;i++)
        gradient_img(i,0) = gradient_img(i,col-1) = Vec3d(0,0,0);
    for (int j=0;j<col;j++)
        gradient_img(0,j) = gradient_img(row-1,j) = Vec3d(0,0,0);
    return ;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------Calculate gradient of original image------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------













//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------Build a mapping from position to equation-------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
void mp_grad(int& cnt,int& col,int& row,int* mask,cv::Point2i mp[],map<int,int>& eq2num)
{
    cnt = 0;
    for (int i=0;i<row;i++)
        for (int j=0;j<col;j++) 
            if (!mask[i*col+j]) 
            {
                eq2num[i*col+j] = cnt;
                mp[cnt] = cv::Point2i(i, j);
                cnt++;
            }
    return ;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------Build a mapping from position to equation-------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------




void blank(int event, int x, int y, int flags, void *ustc)
{

}






int main()
{
    origin_Img_Name = "originImg.png";
    bg_Img_Name = "bg.png";
    selected_bg = 0;
    my_line.clear();
    if (!read_Img(origin_Img_Name,origin_Img)) return 0;
    gradient(origin_Img);
    if (!read_Img(bg_Img_Name,bg_Img)) return 0;
    origin_Img.copyTo(temp2);
    while (1) 
    {
        temp2.copyTo(temp1);
        int o = waitKey(10);
        if (o == 'q') break;
        setMouseCallback("Display window", on_mouse, (void*)&temp2);
        putText(temp1,"("+std::to_string((int)mouse_Position.x)+","+std::to_string((int)mouse_Position.y)+")" , mouse_Position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        imshow( "Display window", temp1/255);
    }
    setMouseCallback("Display window", blank, (void*)&temp2);
    int col = origin_Img.cols;
    int row = origin_Img.rows;
    int mask[row][col];
    printf("%lu %d %d\n",my_line.size(),col,row);
    origin_Img.copyTo(temp1);
    for (int i=0;i<row;i++)
        for (int j=0;j<col;j++) 
            if (IsPointInPolygon(Point2d(j,i),my_line)) temp1(i,j) = Vec3d(0,0,0),mask[i][j] = 0;
                else temp1(i,j) = Vec3d(1,1,1),mask[i][j]=1;
    imshow( "Display window", temp1);
    waitKey(0);
    bg_Img.copyTo(temp2);
    while (1) 
    {
        temp2.copyTo(temp1);
        int o = waitKey(10);
        if (o == 'q') break;
        setMouseCallback("Display window", on_mouse2, (void*)&temp2);
        putText(temp1,"("+std::to_string((int)mouse_Position.x)+","+std::to_string((int)mouse_Position.y)+")" , mouse_Position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        imshow( "Display window", temp1/255);
    }
    setMouseCallback("Display window", blank, (void*)&temp2);
    waitKey(0);

    bg_Position.x -= row/2;
    bg_Position.y += col/2;

    bg_Img.copyTo(temp2);
    for (int i=0;i<row;i++)
        for (int j=0;j<col;j++)
            if (!mask[i][j])
                temp2(i+bg_Position.x,j+bg_Position.y) = origin_Img(i,j);
    imshow( "Display window", temp2/255);
    waitKey(0);

    cv::Point2i mp[col*row];
    int cnt=0;
    map<int,int> eq2num;
    mp_grad(cnt,col,row,(int*)mask,mp,eq2num);
    cout<<cnt<<endl;
    


    Eigen::SparseMatrix<double> pois_A(cnt,cnt);
    Eigen::Matrix<cv::Vec3d,Eigen::Dynamic,Eigen::Dynamic> pois_b(cnt,1);
    for (int i=0;i<row;i++)
        for (int j=0;j<col;j++) 
        {
            int id,num;
            if (!mask[i][j]){
                id = i*col+j;
                num = eq2num[id];
                pois_A.insert(num,num) = 4;
                pois_b(eq2num[id],0) = gradient_img(i,j);
                
                for (int k=0;k<4;k++)
                    if (!mask[i+dx[k]][j+dy[k]]) 
                    {
                        id = (i+dx[k])*col+j+dy[k];
                        pois_A.insert(num,eq2num[id]) = -1;
                    }
                    else pois_b(num,0) += bg_Img(i+dx[k]+bg_Position.x,j+dy[k]+bg_Position.y);

            }
        }
    pois_A.makeCompressed();


    Eigen::VectorXd pois_b_r(cnt),pois_b_g(cnt),pois_b_b(cnt);
    Eigen::VectorXd pois_x_r(cnt),pois_x_g(cnt),pois_x_b(cnt);
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > solver;
    solver.compute(pois_A);
    if(solver.info()!=Eigen::Success) {
      // decomposition failed
      return 0;
    }

    for (int i=0;i<cnt;i++) 
    {
        pois_b_r(i) = pois_b(i,0)(0);
        pois_b_g(i) = pois_b(i,0)(1);
        pois_b_b(i) = pois_b(i,0)(2);
    }

    pois_x_r = solver.solve(pois_b_r);
    pois_x_g = solver.solve(pois_b_g);
    pois_x_b = solver.solve(pois_b_b);

    for (int i=0;i<cnt;i++) 
    {
        pois_b(i,0)(0) = pois_x_r(i);
        pois_b(i,0)(1) = pois_x_g(i);
        pois_b(i,0)(2) = pois_x_b(i);
        bg_Img(mp[i].x+bg_Position.x,mp[i].y+bg_Position.y) = pois_b(i,0);
    }

    imshow( "Display window", bg_Img/255);
    waitKey(0);

    return 0;
}


