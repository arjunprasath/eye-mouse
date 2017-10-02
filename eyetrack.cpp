#include"stdafx.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include<iostream>
using namespace std;IplImage*greenchk(IplImage*imgg);IplImage*cvAndmod(IplImage*im1,IplImage*im2);
using namespace cv;
IplImage *IavgF, *IdiffF, *IprevF, *IhiF, *IlowF;
IplImage *Iscratch, *Iscratch2;
IplImage *Igray1, *Igray2, *Igray3;
IplImage *Ilow1, *Ilow2, *Ilow3;
IplImage *Ihi1, *Ihi2, *Ihi3;
IplImage *Imaskt;
float Icount;

// Function of allocating images
void AllocateImages( IplImage* I ){
	CvSize sz = cvGetSize( I );

	IavgF = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IdiffF = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IprevF = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IhiF = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IlowF = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Ilow1 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ilow2 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ilow3 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi1 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi2 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi3 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	cvZero( IavgF );
	cvZero( IdiffF );
	cvZero( IprevF );
	cvZero( IhiF );
	cvZero( IlowF );
	Icount = 0.0001;	// protect against divid by 0

	Iscratch= cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Iscratch2 = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Igray1 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Igray2 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Igray3 = cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Imaskt = cvCreateImage( sz, IPL_DEPTH_8U, 1 );
	cvZero( Iscratch );
	cvZero( Iscratch2 );
}

// Learn the background statistics for one more frame
void accumulateBackground( IplImage *I ){
	static int first = 1;
	cvCvtScale( I, Iscratch, 1, 0 );
	if( !first ){
		cvAcc( Iscratch, IavgF );
		cvAbsDiff( Iscratch, IprevF, Iscratch2 );
		cvAcc( Iscratch2, IdiffF );
		Icount += 1.0;
	}
	first = 0;
	cvCopy( Iscratch, IprevF );
}

void setHighThreshold( float scale ) {
	cvConvertScale( IdiffF, Iscratch, scale );
	cvAdd( Iscratch, IavgF, IhiF );
	cvSplit( IhiF, Ihi1, Ihi2, Ihi3, 0 );
}

void setLowThreshold( float scale ) {
	cvConvertScale( IdiffF, Iscratch, scale );
	cvAdd( Iscratch, IavgF, IlowF );
	cvSplit( IlowF, Ilow1, Ilow2, Ilow3, 0 );
}

void createModelsfromStats(){
	cvConvertScale( IavgF, IavgF, (double)(1.0/Icount) );
	cvConvertScale( IdiffF, IdiffF, (double)(1.0/Icount) );

	//Make sure diff is always something
	cvAddS( IdiffF, cvScalar( 1.0, 1.0, 1.0), IdiffF );
	setHighThreshold( 7.0);
	setLowThreshold( 6.0 );
}

// Create a mask
void backgroundDiff( IplImage *I, IplImage *Imask ){
	cvCvtScale( I, Iscratch, 1, 0);
	cvSplit( Iscratch, Igray1, Igray2, Igray3, 0 );

	// channel 1
	cvInRange( Igray1, Ilow1, Ihi1, Imask );
	// channel 2
	cvInRange( Igray2, Ilow2, Ihi2, Imaskt );
	cvOr( Imask, Imaskt, Imask );
	// channel 3
	cvInRange( Igray3, Ilow3, Ihi3, Imaskt );
	cvOr( Imask, Imaskt, Imask );
}

void DeallocateImages()
{
	cvReleaseImage( &IavgF );
	cvReleaseImage( &IdiffF );
	cvReleaseImage( &IprevF );
	cvReleaseImage( &IhiF );
	cvReleaseImage( &IlowF );
	cvReleaseImage( &Ilow1 );
	cvReleaseImage( &Ilow2 );
	cvReleaseImage( &Ilow3 );
	cvReleaseImage( &Ihi1 );
	cvReleaseImage( &Ihi2 );
	cvReleaseImage( &Ihi3 );
	cvReleaseImage( &Iscratch );
	cvReleaseImage( &Iscratch2 );
	cvReleaseImage( &Igray1 );
	cvReleaseImage( &Igray2 );
	cvReleaseImage( &Igray3 );
	cvReleaseImage( &Imaskt );
}


int backproject_mode = 0;
int select_object = 0;
int track_object = 0;
int show_hist = 1;
CvPoint origin;
CvRect selection;
CvRect track_window;
CvBox2D track_box;
CvConnectedComp track_comp;
CvHistogram *hist = 0; IplImage*histimg,*mask,*backproject,*hue2,*hue4,*hue3,*hue4t,*hue4g,*hue5;
int hdims = 16;
float hranges_arr[] = {0,180};
float* hranges = hranges_arr; int bin_w;


CvScalar hsv2rgb( float hue )
{
    int rgb[3], p, sector;
    static const int sector_data[][3]=
        {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};
    hue *= 0.033333333333333333333333333333333f;
    sector = cvFloor(hue);
    p = cvRound(255*(hue - sector));
    p ^= sector & 1 ? 255 : 0;

    rgb[sector_data[sector][0]] = 255;
    rgb[sector_data[sector][1]] = 0;
    rgb[sector_data[sector][2]] = p;

    return cvScalar(rgb[2], rgb[1], rgb[0],0);
}


int _tmain(int argc, _TCHAR* argv[])
{
    cvNamedWindow("Image:",1);cvNamedWindow("im2",1);IplImage  *mask1, *mask3;
	cvNamedWindow("im3",1);/*    IplImage *img = cvLoadImage("ip.png");
		
        
img->depth=IPL_DEPTH_8U;
	cvRectangle(img,cvPoint(100,100),cvPoint(200,200),cvScalar(255,0,0),1);

	int height     = img->height;
int width      = img->width;
int step       = img->widthStep/sizeof(uchar);
int channels   = img->nChannels;
uchar* data    = (uchar *)img->imageData;
for(int i=1;i<10;i++)
	for(int j=1;j<10;j++)
{
 //if(data[i*step+j*channels+2] <= 254)
	{data[i*step+j*channels+0] = 254;
data[i*step+j*channels+1] = 0;
data[i*step+j*channels+2] = 0;
}		
	}*/
	IplImage* img2 = NULL;
	IplImage* conimg=NULL;
IplImage* frame = 0;
	IplImage*hue=NULL;
CvCapture* capture = 0;

      

    capture = cvCaptureFromCAM(0);
	CvMemStorage*storage=cvCreateMemStorage(0);
	CvMemStorage*dftstorage=cvCreateMemStorage(0);
	CvMemStorage*minstorage=cvCreateMemStorage(0);
CvSeq*contour=0;CvSeq*hull=0,*defect=0;
int mode = CV_RETR_EXTERNAL;  double fps=0; double loop=0;

//CvSeq* contours2 = NULL;

//CvRect rect = cvBoundingRect( contours2, 0 );

//cvRectangle( bitImage, cvPoint(rect.x, rect.y + rect.height), cvPoint(rect.x + rect.width, rect.y), CV_RGB(200, 0, 200), 1, 8, 0 );


selection=cvRect(300,100,50,50);
	while(1){
	
           loop++;
            frame = cvQueryFrame( capture );
			 fps = cvGetCaptureProperty (
capture,
CV_CAP_PROP_FPS
);        // cout<<fps<<"\n";
            if( !frame )
                break;



			img2=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
		        img2=cvCloneImage(frame);
			cvSmooth(img2,img2,CV_GAUSSIAN);
			//img2=cvLoadImage("hand.png",1);


			/*CvSize sz = cvGetSize( frame );
		mask1 = cvCreateImage( sz, IPL_DEPTH_8U, 1 );
		mask3 = cvCreateImage( sz, IPL_DEPTH_8U, 3 );
		if(loop == 1)
			AllocateImages( frame );
		

		if( loop < 30 ){
			accumulateBackground( frame );
		}else if( loop == 30 ){
			createModelsfromStats();
		}else{
			backgroundDiff( frame, mask1 );

			cvCvtColor(mask1,mask3,CV_GRAY2BGR);
			cvNorm( mask3, mask3, CV_C, 0);
			cvThreshold(mask3, mask3, 100, 1, CV_THRESH_BINARY);
			cvMul( frame, mask3, frame, 1.0 );
			cvShowImage( "Background Averaging", frame );
		}*/







		

cvRectangle(img2,cvPoint(selection.x,selection.y),cvPoint(selection.x+selection.width,selection.y+selection.height),cvScalar(0,255,0,0),4);

       


mode = CV_RETR_CCOMP;
IplImage*hsvimg=cvCreateImage(cvSize(img2->width,img2->height),IPL_DEPTH_8U,3);
cvCvtColor(img2,hsvimg,CV_RGB2HSV);


 backproject = cvCreateImage( cvGetSize(frame), 8, 1 );
hue=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,1);     
hue3=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,3); hue5=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,3);
hue4=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,3); hue4t=cvCreateImage(cvGetSize(hue4),IPL_DEPTH_8U,1);
hue4g=cvCreateImage(cvGetSize(hue4),IPL_DEPTH_8U,1);
//if(loop<60)
//{ hue2=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,3);hue2=greenchk(hsvimg);cout<<"1"<<"\n";}
//cvInRangeS(hsvimg,cvScalar(0,0,100),cvScalar(55,120,255),hue4t);
hue4=greenchk(hsvimg);
IplConvKernel*el= cvCreateStructuringElementEx(3, 3,1,1, CV_SHAPE_ELLIPSE);
cvMorphologyEx(hue4, hue4,hue4, el, CV_MOP_OPEN);

cvDilate(hue4,hue4);
/*if(loop>60)	
{hue3=greenchk(hsvimg);
cvAbsDiff(hue2,hue3,hue4);cout<<"2"<<"\n";}*/
//cvCvtColor(hue2,hue,CV_RGB2GRAY);
/*if(loop>70)
	{hue4=cvAndmod(hue2,hue3);
hue4=cvAndmod(hue4,hue5);cout<<"3"<<"\n";}
}*/
cvCvtColor(hue4,hue4g,CV_RGB2GRAY);
cvThreshold(hue4g,hue4t,120,255,CV_THRESH_BINARY);
 
//hue4g=cvCloneImage(hue4t);
cvFindContours(hue4t, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

					 hull = cvConvexHull2( contour, storage, CV_CLOCKWISE, 0 );
					cvDrawContours(img2, hull, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0), 2, 2, 8);

 defect = cvConvexityDefects( contour, hull, storage );

//CvBox2D box = cvMinAreaRect2( contour, minstorage );

//draw the contour


CvConvexityDefect* defectArray;
 for(;defect;defect = defect->h_next)  
        {  
             int nomdef = defect->total; // defect amount  
       	           
             if(nomdef == 0)  
                 continue;  
                
             // Alloc memory for defect set.     
     //fprintf(stderr,"malloc\n"); 
 
             defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);  
               
             // Get defect set.  
     //fprintf(stderr,"cvCvtSeqToArray\n");  

             cvCvtSeqToArray(defect,defectArray, CV_WHOLE_SEQ);  
   
            // Draw marks for all defects.  

			/* for(;defect;defect = defect->h_next) 
{ 
        int nomdef = defect->total;
        if(nomdef == 0)  
    continue; 
    defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);     
    cvCvtSeqToArray (defect, defectArray, CV_WHOLE_SEQ);*/
 /*   for(int i=0; i<nomdef;i++)
    { 
        cvCircle( img2, *(defectArray[i].end), 5, CV_RGB(255,0,0), -1, 8,0);  
        cvCircle( img2, *(defectArray[i].start), 5, CV_RGB(0,0,255), -1, 8,0); 
        cvCircle( img2, *(defectArray[i].depth_point), 5, CV_RGB(0,255,255), -1, 8,0);        
    }
    
    free(defectArray);
    }*/


            for(int i=0; i<nomdef; i++)  
             {  
                 cvLine(img2, *(defectArray[i].start), *(defectArray[i].depth_point),CV_RGB(0,0,255),1, CV_AA, 0 );  
                 cvCircle(img2, *(defectArray[i].depth_point), 5, CV_RGB(0,255,0), -1, 8,0);  
                 cvCircle(img2, *(defectArray[i].start), 5, CV_RGB(0,255,0), -1, 8,0);  
                 cvLine(img2, *(defectArray[i].depth_point), *(defectArray[i].end),CV_RGB(0,0,255),1, CV_AA, 0 );  
         
	       }  
   
       //  j++;  
                
             // Free memory.         
             free(defectArray);  
         }  
		 

mask=cvCreateImage(cvGetSize(hsvimg),IPL_DEPTH_8U,1);
if(loop>1)
cvSplit(hsvimg,hue,0,0,0);
//cvThreshold(hue,hue2,60,255,CV_THRESH_BINARY);

//cvDilate(hue4,hue4);
//cvErode(hue4,hue4);
//cvSmooth(hue4,hue4);
cvShowImage("im4",hue4);
//cvShowImage("im5",hue4g);
//cvShowImage("im6",hue2);
//cvFindContours(hsvimg, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
//draw the contour
//cvDrawContours(conimg, contour, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0), 2, 2, 8);
//cvShowImage( "im2", hue ); 
if(loop<60)
{
 hist = cvCreateHist( 1, &hdims, CV_HIST_ARRAY, &hranges, 1 );
                histimg = cvCreateImage( cvSize(320,200), 8, 3 );
                cvZero( histimg );

 float max_val = 0.f;
                   cvSetImageROI( hue, selection );
                    cvSetImageROI( mask, selection );
                    cvCalcHist( &hue, hist, 0, mask );
                    cvGetMinMaxHistValue( hist, 0, &max_val, 0, 0 );
                    cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
                    cvResetImageROI( hue );
                   cvResetImageROI( mask );
                    //track_window = selection;
                    //track_object = 1;
	//				cvShowImage("im3",hist->bins);
                   track_window=selection;
				   track_object=1;
				   cvZero( histimg );
                    bin_w = histimg->width / hdims;
                    for(int i = 0; i < hdims; i++ )
                    {
                        int val = cvRound( cvGetReal1D(hist->bins,i)*histimg->height/255 );
                        CvScalar color = hsv2rgb(i*180.f/hdims);
                        cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
                                     cvPoint((i+1)*bin_w,histimg->height - val),
                                     color, -1, 8, 0 );

                    }
}

					cvCalcBackProject( &hue, backproject, hist );
					                cvAnd( backproject, mask, backproject, 0 );
                cvCamShift( backproject, track_window,
                            cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ),
                            &track_comp, &track_box );
                track_window = track_comp.rect;

                if( backproject_mode )
                    cvCvtColor( backproject, img2, CV_GRAY2BGR );
                if( !img2->origin )
                    track_box.angle = -track_box.angle;
                cvEllipseBox( img2, track_box, CV_RGB(255,0,0), 3, CV_AA, 0 );
				if(loop<100)
		selection=cvRect(300,100,50,50);
				//else selection=cvRect(track_box.center.x-30,track_box.center.y-30,60,60);

					cvShowImage( "im2", histimg ); 
				//cvLaplace(backproject,backproject);
					cvSmooth(backproject,backproject,CV_GAUSSIAN);
					cvThreshold(backproject,backproject,120,255,CV_THRESH_BINARY);
					
					//cvErode(backproject,backproject);
					//cvDilate(backproject,backproject);
			
					
	
//cvShowImage( "im2", hue ); 

					cvShowImage( "Image:", img2 );
					cvShowImage( "im3", backproject ); 


					/*while(1){
	
           
            frame = cvQueryFrame( capture );
			 fps = cvGetCaptureProperty (
capture,
CV_CAP_PROP_FPS
);        // cout<<fps<<"\n";
            if( !frame )
				break;
			cvShowImage( "Image2:", frame );	
					}
					*/
cvReleaseImage(&hsvimg);
	cvWaitKey(3);}
	DeallocateImages();
	cvReleaseCapture(&capture);cvReleaseImage(&hue);cvReleaseImage(&mask);cvReleaseImage(&histimg);cvReleaseImage(&hue2);cvReleaseImage(&hue3);cvReleaseImage(&hue4);
	cvReleaseImage(&frame);cvReleaseMemStorage(&storage); cvReleaseImage(&hue4t);cvReleaseImage(&hue4g);

      cvDestroyWindow("Image:");  cvDestroyWindow("im2"); cvDestroyWindow("im3"); cvDestroyWindow("im4"); cvDestroyWindow("im5");  cvDestroyWindow( "Background Averaging"); 
         cvReleaseImage(&img2); 
         cvReleaseImage(&img2); 
         cvReleaseImage(&img2); 
		
		 cvReleaseImage(&conimg);
		

        return 0;
}
IplImage*greenchk(IplImage*imgg)
{IplImage*tempimg; double D;float H,S,V;
tempimg=cvCreateImage(cvGetSize(imgg),IPL_DEPTH_8U,3);
tempimg=cvCloneImage(imgg);
cvCvtColor(tempimg,tempimg,CV_HSV2RGB);
CvScalar c,s;
for(int i=0;i<(imgg->height);i++)//In the 2D array of the img..count till the vertical pixel reaches the height of src
{//cout<<"\n";
for(int j=0;j<(imgg->width);j++)//In the 2D array of the img..count till orizontal pixel reaches the width of src
{
s=cvGet2D(imgg,i,j); 
H=s.val[2];  S=s.val[1];   V=s.val[0];

D=sqrt((H-0)*(H-0)+(S-0)*(S-0)+(V-0)*(V-0)); 
//cout<<D<<"\t";

//if((H>0)&&(H<37)&&(S>0.23)&&(S<0.68))
if((H>0)&&(H<160))
	//&&(D<50))
	{ //ie. if the pixel is predominantly Green
c.val[2]=255;//Set R to 0
c.val[1]=255;//Set G to 255
c.val[0]=255;//Set B to 0
cvSet2D(tempimg,i,j,c); //Change the pixel value of copy img to pure green(G=255 R=0 B=0)
}

else //Set all other pixels in copy to white
{
c.val[2]=0; // Red
c.val[1]=0;// Green
c.val[0]=0;
cvSet2D(tempimg,i,j,c);
}

} }

return tempimg;
}

IplImage*cvAndmod(IplImage*im1,IplImage*im2)
{cvCvtColor(im1,im1,CV_RGB2HSV);cvCvtColor(im2,im2,CV_RGB2HSV);
	IplImage*tempimg;
tempimg=cvCreateImage(cvGetSize(im1),IPL_DEPTH_8U,3);
double D;float H,S,V,H1,S1,V1;
	CvScalar c,s,s1;
for(int i=0;i<(im1->height);i++)//In the 2D array of the img..count till the vertical pixel reaches the height of src
{
for(int j=0;j<(im1->width);j++)//In the 2D array of the img..count till orizontal pixel reaches the width of src
{s1=cvGet2D(im2,i,j); 
s=cvGet2D(im1,i,j); 
H=s.val[2];  S=s.val[1];   V=s.val[0];
H1=s1.val[2];  S1=s.val[1];   V1=s1.val[0];

D=sqrt((H-H1)*(H-H1)+(S-S1)*(S-S1)+(V-V1)*(V-V1)); 

//if((H>0)&&(H<37)&&(S>0.23)&&(S<0.68))
if(D<200)
	//&&(D<50))
	{ //ie. if the pixel is predominantly Green
c.val[2]=0;//Set R to 0
c.val[1]=0;//Set G to 255
c.val[0]=0;//Set B to 0
cvSet2D(tempimg,i,j,c); //Change the pixel value of copy img to pure green(G=255 R=0 B=0)
}

else //Set all other pixels in copy to white
{
c.val[2]=255; // Red
c.val[1]=255;// Green
c.val[0]=255;
cvSet2D(tempimg,i,j,c);
}

} }
return tempimg;}
