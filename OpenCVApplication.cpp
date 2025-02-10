// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <queue>
#include <opencv2/video/tracking.hpp>

#define MAX_HUE 256
//variabile globale
int histG_hue[MAX_HUE]; // histograma globala / cumulativa 
#define FILTER_HISTOGRAM 1


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the "diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void binarizare_globala(Mat &img) {

		float minn = 256.0;
		float maxx = 0.0;
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				if (minn > (float)img.at<uchar>(i, j))
					minn = (float)img.at<uchar>(i, j);
				else
					if (maxx < (float)img.at<uchar>(i, j))
						maxx = (float)img.at<uchar>(i, j);
			}
		float treshhold = (minn + maxx) / 2;
		float treshhold_vechi = 0.0;
		float mediamica = 0.0;
		int nrmic = 0;
		int nrmare = 0;
		float mediamare = 0.0;
		float dif = 1.0;
		while (dif > 0.1)
		{
			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					if ((float)img.at<uchar>(i, j) < treshhold)
					{
						mediamica = mediamica + (float)img.at<uchar>(i, j);
						nrmic++;
					}
					else
					{
						mediamare = mediamare + (float)img.at<uchar>(i, j);
						nrmare++;
					}
				}
			mediamare = mediamare / nrmare;
			mediamica = mediamica / nrmic;
			nrmare = 0;
			nrmic = 0;
			treshhold_vechi = treshhold;
			treshhold = (mediamare + mediamica) / 2;
			dif = treshhold - treshhold_vechi;
		}

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				if ((float)img.at<uchar>(i, j) < treshhold)
					img.at<uchar>(i, j) = 255;
				else
					img.at<uchar>(i, j) = 0;
			}
		imshow("binarizata automat", img);
	
}

//lab 2 sapt 1
void lab1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname,IMREAD_COLOR);
		imshow("image", src);


		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		Mat channels[3];
		split(hsvImg, channels);
		
		Mat H = channels[0] * (256.0 / 180);
		Mat S = channels[1];
		Mat V = channels[2];


		int* histH = (int*)calloc(256, sizeof(int));
		for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
			{
				uchar pixel = H.at<uchar>(i, j);
				histH[pixel]++;
			}
		
		int* histS = (int*)calloc(256, sizeof(int));
		for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
			{
				uchar pixel = S.at<uchar>(i, j);
				histS[pixel]++;
			}


		int* histV= (int*)calloc(256, sizeof(int));
		for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
			{
				uchar pixel = V.at<uchar>(i, j);
				histV[pixel]++;
			}
		
		showHistogram("Histograma H", histH, 256, 256);
		showHistogram("Histograma V", histV, 256, 256);
		showHistogram("Histograma S", histS, 256, 256);
		
		Mat Hcopy=H.clone();
		binarizare_globala(Hcopy);
		imshow("H BEFORE", H);

		for (int i=0; i<H.rows;i++)
			for (int j = 0; j < H.cols; j++)
			{
				if (H.at<uchar>(i, j) < 123)
					H.at<uchar>(i, j) = 255;
				else
					H.at<uchar>(i, j) = 0;
			}
		imshow("H binarizat manual", H);

		
		
		
		
		waitKey(0);
	}

	//binarize dupa un prag


}

//lab 3 sapt 2
void L3()
{
	Mat src;
	Mat hsv;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		// Componenta de culoare Hue a modelului HSV
		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		Mat channels[3];
		split(hsvImg, channels);
		Mat H = channels[0] * (256.0 / 180);
		imshow("H BEFORE", H);
		
		// 3.2.1. clasificarea pixelilor:
		Mat dst = Mat(height, width, CV_8UC1);
		float k = 2.5, mean = 16, std = 5;
		int ci = 0, ri = 0; //centre de masa
		int surface = 0;
		// if (apartine interval) => pixel obiect else pixel fundal
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int hue = H.at<uchar>(i, j);
				//(hue_mean-k*hue_std) .. (hue_mean+k*hue_std)
				if ((mean - k * std) < hue && hue < (mean + k * std)) {
					dst.at<uchar>(i, j) = 255;
					ci = ci + j;
					ri = ri + i;
					surface++;
				}
				else
					dst.at<uchar>(i, j) = 0;
			}
		}
		ri = ri / surface;
		ci = ci / surface;

		imshow("Segmentarea", dst);

		//3.2.2. Postprocesarea imaginii segmentate
		
		// 2 x eroziuni + 4x dilatari + 2x eroziuni 
		// creare element structural de dimensiune 3x3 de tip patrat (V8)
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

		//eroziune cu acest element structural (aplicata 2x)
		erode(dst, dst, element, Point(-1, -1), 2);
		
		// dilatare cu acest element structural (aplicata 4x)
		dilate(dst, dst, element, Point(-1, -1), 4);

		//eroziune cu acest element structural (aplicata 2x)
		erode(dst, dst, element, Point(-1, -1), 2);

		imshow("Postprocesare", dst);

		// 3.2.3.Extragerea conturului mainii
		Labeling("Contur", dst, false);

		// 3.2.4. Desenarea axei de alungire a mainii
		float aux1 = 0, aux2 = 0, aux3 = 0, slope;
		for (int i = 0; i < dst.rows; i++)
		{
			for (int j = 0; j < dst.cols; j++)
			{
				if (255 == dst.at<uchar>(i, j))
				{
					aux1 = aux1 + (i - ri) * (j - ci);
					aux2 = aux2 + (j - ci) * (j - ci);
					aux3 = aux3 + (i - ri) * (i - ri);
				}
			}
		}

		slope = atan2((2 * aux1), (aux2 - aux3)) / 2.0;
		printf("%f", slope);
		
		
		line(dst, Point(ci, ri), Point((int)(ci + 300 * cos(slope)), (int)(ri + 300 * sin(slope))), Vec3b(100, 100, 0));
		line(dst, Point(ci, ri), Point((int)(ci + 300 * cos(slope+PI)), (int)(ri + 300 * sin(slope+PI))), Vec3b(100, 100, 0));
		imshow("Slope", dst);

		waitKey(0);
	}
}

//lab 4 sapt 3

//(a) Regiuni uniforme obtinute prin cresterea unui bloc/seed prin unirea altor pixeli sau blocuri de pixeli
bool isInside(Mat img, int i, int j) {
	if ((i < img.rows) && (j < img.cols) && (i >= 0) && (j >= 0))
		return true;
	else
		return false;
}

uchar Hue_averge(int x, int y, Mat H) {
	int L[] = { -1, 0 , +1 };
	int R[] = { -1, 0 , +1 };
	uchar sum = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			int in = x + L[i];
			int jn = y + R[j];
			if (isInside(H,in,jn))
				sum += H.at<uchar>(in , jn);
		}
	}

	return (uchar)sum / 9;
}

void L4(int event, int x, int y, int flags, void* param) {
	Mat* src1 = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		// 2. Se alege un punct de start ales cu ajutorul mouse-ului .
		printf("Pos(x,y): %d,%d \n", x, y);
		Point p = Point(x, y);

		Mat H = (*src1); //imagine sursa
		int height = H.rows;
		int width = H.cols;

		// 3. Algorithmul:

		//A. Se va aloca o matrice de etichete labels de dimensiunea imaginii.
		Mat labels = Mat::zeros(H.size(), CV_8UC1);
		int di[8] = { -1, 0, 1, 0, -1, -1, 1, 1 };
		int dj[8] = { 0, -1, 0, 1, 1, -1, 1, -1 };

		queue <Point> que;
		uchar label = 1; //ethiceta curenta
		int N = 1; // numarul de pixeli din regiune
		que.push(Point(x, y)); // adauga element (seed point) in coada
		// acesta primeste eticheta label
		labels.at<uchar>(x, y) = label;
		uchar Hue_avg = Hue_averge(x, y, H);
		
		float T =  Hue_avg*2.2;
		while (!que.empty())
		{ 
			// Retine poz. celui mai vechi element din coada
			Point oldest = que.front();
			que.pop(); // scoate element din coada
			int xx = oldest.x; // coordonatele lui
			int yy = oldest.y;
			// Pentru fiecare vecin al pixelului (xx, yy) ale carui coordonate sunt in interiorul imaginii:
			for (int k = 0; k < 8; k++) {
				Point n = Point(xx + di[k], yy + dj[k]);
				if (isInside(H, xx + di[k], yy + dj[k]) == true)
				{
					uchar neighbor = H.at<uchar>(n.x, n.y);
					// Daca abs(hue(vecin) – Hue_avg)<T si labels(vecin) == 0
					if (abs(neighbor - Hue_avg) < (uchar)T && labels.at<uchar>(n.x, n.y) == 0) {
						//  adauga pixelul (i,j) in lista FIFO la pozitia top (este adaugat la regiunea curenta)
						que.push(n);
						//  pixelul (i,j) primeste eticheta k: labels(i,j)=k
						labels.at<uchar>(n.x, n.y) = label;
						Hue_avg = (N * Hue_avg + neighbor) / (N + 1);
						//  incrementeaza N
						N++;
					}
				}
			}
		}
		imshow("Hue", H);

		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				
				if (labels.at<uchar>(i, j) == 1) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}


		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

		//eroziune cu acest element structural (aplicata 2x)
		erode(dst, dst, element, Point(-1, -1), 3);

		// dilatare cu acest element structural (aplicata 4x)
		dilate(dst, dst, element, Point(-1, -1), 2);

		imshow("Result", dst);

		waitKey(0);
	}
}

void MouseCallL4()
{
	Mat src;
	Mat hsv;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		// Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		//Creare fereastra pt. afisare
		namedWindow("src", 1);
		// Componenta de culoare Hue a modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		// definire pointeri la matricea (8 biti/pixeli) folosita la stocarea
		// componentei individuale H
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV); // conversie RGB -> HSV
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		Mat channels[3];
		split(hsv, channels);

		H = channels[0] * (256.0 / 180);
		// Asociere functie de tratare a avenimentelor MOUSE cu ferestra curenta
		// Ultimul parametru este matricea H (valorile compunentei Hue)
		setMouseCallback("src", L4, &H);
		imshow("src", src);
		// Wait until user press some key
		waitKey(0);
	}
}

//lab 5 sapt 4

// 1. + 2.
void findCorners() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = src.clone();
		int height = src.rows;
		int width = src.cols;

		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

		// 1. img to grey scale : clone
		Mat Src_image_gray;
		cvtColor(src, Src_image_gray, CV_BGR2GRAY);
		//imshow("greyscale", Src_image_gray);

		// 2. Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(Src_image_gray, Src_image_gray, Size(5, 5), 0, 0);

		// 3. Apoi good features
			// Parametrii functiei
			// Lista/vector care va contine coordonatele (x,y) ale colturilor detectate (output)
		vector<Point2f> corners;
		// Nr. maxim de colturi luate in considerare. Daca nr. de colturi > maxCorners se vor considera cele cu raspuns R maxim
		int maxCorners = 100;
		// Factor cu care se multiplica masura de calitate a celui mai bun colt (val. proprie minima) pt.metoda Shi - Tomasi respectiv valoarea functiei de raspuns R(Harris)
		double qualityLevel = 0.01;
		// Distana euclidiana minima dintre 2 colturi returnate
		double minDistance = 10;
		// Dimensiunea ferestrei w in care se calculeaza matricea de autocorelatie 
		int blockSize = 3; // 2,3, ...
			// Selectia metodei de detectie: Harris (true) sau Shi-Tomasi (false).
		bool useHarrisDetector = true;
		// Factorul k (vezi documentatia curs)
		double k = 0.04;
		// Apel functie
		goodFeaturesToTrack(Src_image_gray,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(), //masca pt. ROI - optional
			blockSize,
			useHarrisDetector,
			k);

		cornerSubPix(Src_image_gray, corners, winSize, zeroZone, criteria);


		//	4. Apoi aplicam cercuri pe imagine
		int r = 5;
		int nr_corners = corners.size();
		// Write to file corners:
		std::ofstream outFile("corners.txt");
		if (outFile.is_open()) {
			outFile << std::fixed << std::setprecision(2);


			for (int i = 0; i < nr_corners; i++) {
				outFile << "(" << corners[i].x << ","<< corners[i].y<< ")" << std::endl;
			}

			outFile.close();
			std::cout << "Corners have been written to the file." << std::endl;
		}
		else {
			std::cerr << "Unable to open the file for writing." << std::endl;
		}
		// Draw circles:
		for (int i = 0; i < nr_corners; i++) {
			circle(dst, corners[i], r, Scalar(0, 255, 0), -1, 8, 0);
		}

		//	5. SHOW IMG
		imshow("corners", dst);

		waitKey(0);
	}
}

// 4.
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;
const char* corners_window = "Corners detected";

void cornerHarris_demo(int, void*)
{
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	Mat dst = Mat::zeros(src.size(), CV_32FC1); //- va contine functia de raspuns R(x, y)
	cornerHarris(src_gray, dst, blockSize, apertureSize, k);
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //- aceasta valoare se normalizeaza in interv. 0..255 si se pune intr - o imagine grayscale(1 channel) : dst_norm_scaled
	convertScaleAbs(dst_norm, dst_norm_scaled);//- Zonele netede din imagine(R(x, y) 0) sunt mapate in nuante de gri
	
	
	
	int d = 5;
	int w = 2 * d + 1;
	int di[8] = { -1, -1,-1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			bool flag = true; 
			if ((int)dst_norm.at<float>(i, j) > thresh) //- Punctele de muchie(R(x, y) < 0 sunt mapate in nuante inchise(negru)- Punctele de colt(R(x, y) > 0 sunt mapate in nuante deschise(alb)
			{

				for (int x = 0; x < w; x++)  // parcurgem sablonul
				{
					for (int y = 0; y < w; y++)
					{
						int	auxx = i + x - (w / 2);
						int	auxy = j + y - (w / 2);
						//int vSrc = dst.at<uchar>(auxx, auxy);
						if (dst_norm.at<float>(auxx,auxy) > dst_norm.at<float>(i, j)) flag = false;
					}
				}

				if (flag) circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	namedWindow(corners_window);
	imshow(corners_window, dst_norm_scaled);
}

void findCorners2() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = src.clone();
		int height = src.rows;
		int width = src.cols;

		// 1. img to grey scale : clone
		Mat Src_image_gray;
		cvtColor(src, Src_image_gray, CV_BGR2GRAY);
		src_gray = Src_image_gray;
		//imshow("greyscale", Src_image_gray);

		// 2. Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(Src_image_gray, Src_image_gray, Size(5, 5), 0, 0);

		cornerHarris_demo(0, 0);


		waitKey(0);
	}
}

// 5.
void findCornersVideo() {

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		
		//findCorners();
		Mat src = frame;
		Mat dst = src.clone();
		int height = src.rows;
		int width = src.cols;

		// 1. img to grey scale : clone
		Mat Src_image_gray;
		cvtColor(src, Src_image_gray, CV_BGR2GRAY);
		//imshow("greyscale", Src_image_gray);

		// 2. Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		GaussianBlur(Src_image_gray, Src_image_gray, Size(5, 5), 0, 0);

		// 3. Apoi good features
			// Parametrii functiei
			// Lista/vector care va contine coordonatele (x,y) ale colturilor detectate (output)
		vector<Point2f> corners;
		// Nr. maxim de colturi luate in considerare. Daca nr. de colturi > maxCorners se vor considera cele cu raspuns R maxim
		int maxCorners = 100;
		// Factor cu care se multiplica masura de calitate a celui mai bun colt (val. proprie minima) pt.metoda Shi - Tomasi respectiv valoarea functiei de raspuns R(Harris)
		double qualityLevel = 0.01;
		// Distana euclidiana minima dintre 2 colturi returnate
		double minDistance = 10;
		// Dimensiunea ferestrei w in care se calculeaza matricea de autocorelatie 
		int blockSize = 3; // 2,3, ...
			// Selectia metodei de detectie: Harris (true) sau Shi-Tomasi (false).
		bool useHarrisDetector = true;
		// Factorul k (vezi documentatia curs)
		double k = 0.04;
		// Apel functie
		goodFeaturesToTrack(Src_image_gray,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(), //masca pt. ROI - optional
			blockSize,
			useHarrisDetector,
			k);



		//	4. Apoi aplicam cercuri pe imagine
		int r = 5;
		int nr_corners = corners.size();
		// Draw circles:
		for (int i = 0; i < nr_corners; i++) {
			circle(dst, corners[i], r, Scalar(0, 255, 0), -1, 8, 0);
		}

		//	5. SHOW IMG
		imshow("corners", dst);
		///

		imshow("source", frame);
		imshow("gray", grayFrame);

		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


//lab 6 sapt 5

void BackgroundSubtraction() {

	VideoCapture cap("Videos/laboratory.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam

	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat frame, gray; //current frame: original and gray
	Mat backgnd; // background model
	Mat diff; //difference image: |frame_gray - bacgnd|
	Mat dst; //output image/frame
	char c;
	int frameNum = -1; //current frame counter
	int method = 0;
	// method =
	// 1 - frame difference
	// 2 - running average
	// 3 - running average with selectivity
	const unsigned char Th = 25;
	const double alpha = 0.05;
	//citim metoda:
	printf("metoda:");
	scanf("%d", &method);

	for (;;) {
		cap >> frame; // achizitie frame nou
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		
		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame); // daca este primul cadru se afiseaza doar sursa
		
		// 1. Convertim la grey scale:
		cvtColor(frame, gray, CV_BGR2GRAY);
		// 2. Optional puteti aplica si un FTJ Gaussian
		GaussianBlur(gray, gray, Size(5, 5), 0, 0);

		// 3. Se initializeaza matricea / imaginea destinatie pentru fiecare frame
		dst =  Mat::zeros( gray.size(), gray.type() );

		const int channels_gray = gray.channels();
		// 4. restrictionam utilizarea metodei doar pt. imagini grayscale cu un canal (8 bit / pixel)
		if (channels_gray > 1)
			return;


		if (frameNum > 0) // daca nu este primul cadru
		{
			//------ SABLON DE PRELUCRARI PT. METODELE BACKGROUND SUBTRACTION -------
			// 5. Calcul imagine diferenta dintre cadrul current (gray) si fundal (backgnd)
			// Rezultatul se pune in matricea/imaginea diff
			// Backgnd este mereu primul cadru pentru al doilea cadru
			absdiff(gray, backgnd, diff);
			// 6. Modelarea pixelilor de fundal: Se actualizeaza matricea/imaginea model a fundalului (backgnd) conform celor 3 metode:
			if (method == 1) {
				// met 1: 
				// backgnd ul devine cadrul curent 
				backgnd = gray.clone();
			}
			else if(method == 2){
				// met 2 + 3 : 
				// 2. running average
				// calculam backgrnd-ul cadrului urmator cu media ponderata:
				addWeighted(gray, alpha, backgnd, 1.0-alpha, 0, backgnd);
			}
			
			// 7. Binarizarea matricii diferenta (pt. toate metodele):
			// Se parcurge sistematic matricea diff
			// current pixel is forground (object) -> color in white
			for(int i= 0;i<diff.rows;i++)
				for (int j = 0; j < diff.cols; j++) {
					// daca valoarea pt. pixelul current 
					if (diff.at<uchar>(i, j) > Th)
						dst.at<uchar>(i, j) = 255; // current pixel is forground (object) -> color in white
						// selective running average -> no change to the background pixel model value
					else if (method == 3) {
						//metoda 3: 
						// current pixel is background -> update background model for the current pixel
						backgnd.at<uchar>(i, j) =
							alpha * gray.at<uchar>(i, j) + (1.0 - alpha) * backgnd.at<uchar>(i, j);
					}
				}

			// 8. eroziunii + dilatari:
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			erode(dst, dst, element, Point(-1, -1), 2);
			dilate(dst, dst, element, Point(-1, -1), 2);

			
			//-------------------------------------------------------------------------
			// 9. Afiseaza imaginea sursa si destinatie
			imshow("sursa", frame); // show source
			imshow("dest", dst); // show destination
			// Plasati aici codul pt. vizualizarea oricaror rezultate intermediare
			imshow("diff", diff); 
			imshow("background", backgnd);
			// Ex: afisarea intr-o fereastra noua a imaginii diff
		}
		else // daca este primul cadru, modelul de fundal este chiar el
			backgnd = gray.clone();
		// Conditia de avansare/terminare in cilului for(;;) de procesare
		c = cvWaitKey(0); // press any key to advance between frames
		//for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n");
			break; //ESC pressed
		}
	}
}


//lab 7 sapt 6


//Calculul fluxului optic prin algoritmul Horn-Schunk (iterativ)
void calcOpticalFlowHS(const Mat& prev, const Mat& crnt, float lambda, int n0, Mat& flow)
{
	// 1. Calculam Ex,Ey,Et,vx,vy
	Mat vx = Mat::zeros(crnt.size(), CV_32FC1); // matricea comp. x a fluxului optic
	Mat vy = Mat::zeros(crnt.size(), CV_32FC1); // matricea comp. y a fluxului optic
	Mat Et = Mat::zeros(crnt.size(), CV_32FC1); // derivatele temporale
	Mat Ex, Ey; // Matricele derivatelor spatiale (gradient)
	// Calcul componenta orizontala a gradientului
	Sobel(crnt, Ex, CV_32F, 1, 0);
	// Calcul componenta verticala a gradientului
	Sobel(crnt, Ey, CV_32F, 0, 1);
	// Calcul derivata temporala
	Mat prev_float, crnt_float; // matricile imaginii crnt sip rev se convertesc in float
	prev.convertTo(prev_float, CV_32FC1);
	crnt.convertTo(crnt_float, CV_32FC1);
	Et = crnt_float - prev_float;
	// 2. lambda (ex. lambda = 10) si n0 (n0 = 8) sunt date ca argumente.
	// 3. Pentru n = 1 .. n0 :
	// Se parcurge imaginea.Pentru fiecare pixel p se calculeaza valorile medii ale vx si vy(din vecinii de pe directiile cardinale :
	for (int n = 1; n < n0; n++) {
		for (int i = 1; i < crnt.rows-1; i++)
			for (int j = 1; j < crnt.cols-1; j++) {
				float _vx = 1 / 4 * (vx.at<float>(i - 1, j) + vx.at<float>(i + 1, j) + vx.at<float>(i, j - 1) + vx.at<float>(i, j + 1));
				float _vy = 1 / 4 * (vy.at<float>(i - 1, j) + vy.at<float>(i + 1, j) + vy.at<float>(i, j - 1) + vy.at<float>(i, j + 1));
				float alpha = lambda * ((Ex.at<float>(i, j) * _vx + Ey.at<float>(i, j) * _vy + Et.at<float>(i, j)) / (1 + lambda * (Ex.at<float>(i, j) * Ex.at<float>(i, j) + Ey.at<float>(i, j) * Ey.at<float>(i, j))));
				vx.at<float>(i, j) = _vx - alpha * Ex.at<float>(i, j);
				vy.at<float>(i, j) = _vy - alpha * Ey.at<float>(i, j);
			}
	}
	// Compune comp. x si y ale fluxului optic intr-o matrice cu elemente de tip Point2f
	flow = convert2flow(vx, vy);
	// Vizualizare rezultate intermediare:
	// gradient,derivata temporala si componentele vectorilor de miscare sub forma unor
	// imagini grayscale obtinute din matricile de tip float prin normalizare
	Mat Ex_gray, Ey_gray, Et_gray, vx_gray, vy_gray;
	normalize(Ex, Ex_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Ey, Ey_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Et, Et_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vx, vx_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vy, vy_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	imshow("Ex", Ex_gray);
	imshow("Ey", Ey_gray);
	imshow("Et", Et_gray);
	imshow("vx", vx_gray);
	imshow("vy", vy_gray);
}

void lab7HS()
{
	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat flow; // flow - matrix containing the optical flow vectors/pixel
	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");

	int frameNum = -1; //current frame counter
	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
	// la cate un fisier bitmap din secventa
	{
		float lambda = 8;
		int n0 = 8;
		crnt = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;
		if (frameNum > 0) // not the first frame
		{
			// Horn-Shunk
			double t = (double)getTickCount();
			//calcOpticalFlowHS(prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow);
			calcOpticalFlowHS(prev, crnt, lambda, n0, flow);
			// Stop the proccesing time measure
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
			showFlow("rezultat", prev, flow, 1, 1.5, true, true, false);
		}
		// store crntent frame as previos for the next cycle
		prev = crnt.clone();
		int c = cvWaitKey(0); // press any key to advance between frames

							  //for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed
		}
	}

}

void CalcFlowLK(const Mat& prev, const Mat& crnt)
{
	vector<Point2f> prev_pts; // vector of 2D points with previous image features
	vector<Point2f> crnt_pts;// vector of 2D points with current image (matched) features
	vector<uchar> status; // output status vector: 1 if the wlow for the corresponding feature was found. 0 otherwise
	vector<float> error; // output vector of errors; each element of the vector is set to an error for the corresponding feature
	Size winSize = Size(21, 21); // size of the search window at each pyramid level - deafult (21,21)
	int maxLevel = 3; // maximal pyramid level number - deafult 3
	//parameter, specifying the termination criteria of the iterative search algorithm
	// (after the specif
	// ied maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon
	// deafult 30, 0.01
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
	int flags = 0;
	double minEigThreshold = 1e-4;

	vector<Point2f> corners;
	// Nr. maxim de colturi luate in considerare. Daca nr. de colturi > maxCorners se vor considera cele cu raspuns R maxim
	int maxCorners = 100;
	// Factor cu care se multiplica masura de calitate a celui mai bun colt (val. proprie minima) pt.metoda Shi - Tomasi respectiv valoarea functiei de raspuns R(Harris)
	double qualityLevel = 0.01;
	// Distana euclidiana minima dintre 2 colturi returnate
	double minDistance = 10;
	// Dimensiunea ferestrei w in care se calculeaza matricea de autocorelatie 
	int blockSize = 3; // 2,3, ...
		// Selectia metodei de detectie: Harris (true) sau Shi-Tomasi (false).
	bool useHarrisDetector = true;
	// Factorul k (vezi documentatia curs)
	double k = 0.04;
	// Apel functie
	goodFeaturesToTrack(prev, prev_pts,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(), //masca pt. ROI - optional
		blockSize,
		useHarrisDetector,
		k);


	calcOpticalFlowPyrLK(prev, crnt, prev_pts, crnt_pts, status, error, winSize, maxLevel, criteria);

	showFlowSparse("Dst", prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
}

void lab7LK()
{
	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat flow; // flow - matrix containing the optical flow vectors/pixel
	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");

	int frameNum = -1; //current frame counter
	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
	// la cate un fisier bitmap din secventa
	{
		float lambda = 8;
		int n0 = 8;
		crnt = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;
		if (frameNum > 0) // not the first frame
		{
			// Horn-Shunk
			double t = (double)getTickCount();
			//calcOpticalFlowHS(prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow);
			CalcFlowLK(prev, crnt);
			// Stop the proccesing time measure
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
			showFlow("rezultat", prev, flow, 1, 1.5, true, true, false);
		}
		// store crntent frame as previos for the next cycle
		prev = crnt.clone();
		int c = cvWaitKey(0); // press any key to advance between frames

							  //for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed
		}
	}

}

// laborator 8 sapt 7


void lab8()
{
	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat flow; // flow - matrix containing the optical flow vectors/pixel
	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");
	int frameNum = -1; //current frame counter
	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
	// la cate un fisier bitmap din secventa
	{
		crnt = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;
		if (frameNum > 0) // not the first frame
		{
			//. . . .
			// functii de procesare (calcul flux optic) si afisare
			double t = (double)getTickCount(); // Get the crntent time [s]
			// . . . insert here the processing functions / code // Get the crntent time again and compute the time difference [s]
			int winSize = 15;
			float  minVel = 0.9;
			calcOpticalFlowFarneback(prev, crnt, flow, minVel, 3, winSize, 10, 7, 1.5, 0); //slower but more accurate
			makeColorwheel(); // initaializes the colorwhel for the colorcode module
			make_HSI2RGB_LUT();
			showFlowDense("img", crnt, flow, 0.5, true);
			t = ((double)getTickCount() - t) / getTickFrequency();
			// Print (in the console window) the processing time in [ms]

			int hist_dir[360] = {0};

			int pi = 3.14;
			printf("%d - %.3f [ms]\n", frameNum, t*1000);
			for (int r = 0; r < flow.rows; r++) {
				for (int c = 0; c < flow.cols; c++) {
					Point2f f = flow.at<Point2f>(r, c); // vectorul de miscare in punctual (r,c)
					// vectorul de miscare al punctului se considera cu originea in imaginea trecuta (prev)
					// si varful in imaginea curenta (crnt) –> se iau valorile lui din vectorul flow cu minus !
					float dir_rad = pi + atan2(-f.y, -f.x); //directia vectorului in radiani
					int dir_deg = dir_rad * 180 / pi;
					dir_deg = dir_deg % 360;
					if (dir_deg >= 0 && dir_deg < 360 && dir_deg >= minVel)
						hist_dir[dir_deg]++;
				}
			}

			showHistogram("Hist", hist_dir, 360, 200, true);
			// 200 [pixeli] = inaltimea ferestrei de afisare a histogramei
			showHistogramDir("HistDir", hist_dir, 360, 200, true);

			//. . .
		}
		// store crntent frame as previos for the next cycle
		prev = crnt.clone();
		int c = cvWaitKey(0); // press any key to advance between frames //for continous play use cvWaitKey( delay > 0) 
		if (c == 27) { // press ESC to exit
			printf("ESC pressed - playback finished\n\n"); 
			break; //ESC pressed 
		}
	}

}


// laborator 9 sapt 8


CascadeClassifier face_cascade; // cascade clasifier object for face
CascadeClassifier eyes_cascade; // cascade clasifier object for eyes
CascadeClassifier nose_cascade; // cascade clasifier object for eyes
CascadeClassifier mouth_cascade; // cascade clasifier object for eyes

void FaceDetectandDisplayMouthNose(const string& window_name, Mat frame,
	int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], Scalar(0, 255, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)

		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}


		Rect nose_rect; //nose is the 40% ... 75% height of the face
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4 * faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35 * faces[i].height;

		Mat nose_ROI = frame_gray(nose_rect);
		std::vector<Rect> nose;

		nose_cascade.detectMultiScale(nose_ROI, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize / 5, minFaceSize / 5));

		for (int j = 0; j < nose.size(); j++) {
			Point center(nose_rect.x + nose[j].x + nose[j].width * 0.5,
				nose_rect.y + nose[j].y + nose[j].height * 0.5);

			int radius = cvRound((nose[j].width + nose[j].height) * 0.25);
			circle(frame, center, radius, Scalar(0, 0, 255), 4, 8, 0);
		}


		Rect mouth_rect; //mouth is in the 70% ... 99% height of the face
		mouth_rect.x = faces[i].x;
		mouth_rect.y = faces[i].y + 0.7 * faces[i].height;
		mouth_rect.width = faces[i].width;
		mouth_rect.height = 0.29 * faces[i].height;
		std::vector<Rect> mouth;
		Mat mouth_ROI = frame_gray(mouth_rect);

		mouth_cascade.detectMultiScale(mouth_ROI, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize / 4, minFaceSize / 4));

		for (int j = 0; j < mouth.size(); j++) {
			Point center(mouth_rect.x + mouth[j].x + mouth[j].width * 0.5,
				mouth_rect.y + mouth[j].y + mouth[j].height * 0.5);

			int radius = cvRound((mouth[j].width + mouth[j].height) * 0.25);

			circle(frame, center, radius, Scalar(255, 51, 153), 4, 8, 0);
		}
	}
	imshow(window_name, frame);
	waitKey();

}


void face_detection_mouth_nose() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!mouth_cascade.load(mouth_cascade_name))
	{
		printf("Error loading mouth cascades !\n");
		return;
	}

	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}

	Mat src;
	Mat dst;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;
		FaceDetectandDisplayMouthNose("Dst", dst, minFaceSize, minEyeSize);
	}
}

void FaceDetectandDisplay2(const string& window_name, Mat frame, int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		// get the center of the face
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		// draw circle around the face
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);

	}
	imshow(window_name, frame); //-- Show what you got


}

void FaceDetectandDisplayVideo(const string& window_name, Mat frame, int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		// get the center of the face
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		// draw circle around the face
		ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0,
			360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face (rectangular ROI), detect the eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));
		for (int j = 0; j < eyes.size(); j++)
		{
			// get the center of the eye
		   //atentie la modul in care se calculeaza pozitia absoluta a centrului ochiului 
				// relativa la coltul stanga-sus al imaginii:
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
			// draw circle around the eye
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);


		}
	}
	imshow(window_name, frame); //-- Show what you got


}

void lab9_3() {

	VideoCapture cap("Videos/Megamind.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{

		Mat src, dst;
		dst = frame.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5; // conform proprietatilor antropomorfice alefetei(idem pt.gura si nas)

		String face_cascade_name = "haarcascade_frontalface_alt.xml";
		String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

		if (!face_cascade.load(face_cascade_name))
		{
			printf("Error loading face cascades !\n");
			return;
		}
		if (!eyes_cascade.load(eyes_cascade_name))
		{
			printf("Error loading eyes cascades !\n");
			return;
		}

		double t = (double)getTickCount(); // Get the current time [s]
		FaceDetectandDisplay2("WIN_DST", dst, minFaceSize, minEyeSize);
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);


		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}

}

void lab9_4() {

	Mat src, dst;
	src = imread("Images/Face/BioID_0244.bmp", CV_LOAD_IMAGE_COLOR);
	dst = src.clone();
	int minFaceSize = 30;
	int minEyeSize = minFaceSize / 5; // conform proprietatilor antropomorfice alefetei(idem pt.gura si nas)

	String face_cascade_name = "lbpcascade_frontalface.xml";

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}

	FaceDetectandDisplay2("WIN_DST", dst, minFaceSize, minEyeSize);
	waitKey(0);
}


// laborator 10 sapt 9
Rect FaceDetectand(const string& window_name, Mat frame, int minFaceSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	Mat faceROI[100];
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], Scalar(0, 255, 255), 4, 8, 0);
		faceROI[i] = frame_gray(faces[i]);
		std::vector<Rect> eyes;
	}
	//imshow(window_name, frame); //-- Show what you got
	return faces[0];
}


typedef struct { double arie; double xc; double yc; } mylist;

void lab10() {

	VideoCapture cap("Videos/test_msv1_short.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam

	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat frame, gray; //current frame: original and gray
	Mat backgnd; // background model
	Mat diff; //difference image: |frame_gray - bacgnd|
	Mat dst; //output image/frame
	char c;
	int frameNum = -1; //current frame counter
	const unsigned char Th = 25;
	const double alpha = 0.05;

	
	int x = 0;
	for (;;) {
		cap >> frame; // achizitie frame nou
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		bool clipit = false;
		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame); // daca este primul cadru se afiseaza doar sursa
		Mat src = frame.clone();
		// Prepro: Convertim la grey scale:
		cvtColor(frame, gray, CV_BGR2GRAY);

		// 1. Detectare faciala:
		String face_cascade_name = "haarcascade_frontalface_alt.xml";
		if (!face_cascade.load(face_cascade_name)){printf("Error loading face cascades !\n");return;}
		int minFaceSize = 30;
		Rect faceROI = FaceDetectand("WIN_DST", frame, minFaceSize);

		// 2. Background Subtraction
	
			// 2.2. Optional puteti aplica si un FTJ Gaussian
		GaussianBlur(gray, gray, Size(5, 5), 0, 0);

			// 2.3. Se initializeaza matricea / imaginea destinatie pentru fiecare frame
		dst = Mat::zeros(gray.size(), CV_8UC1); // Ensure single channel (grayscale)

		const int channels_gray = gray.channels();

			// 2.4. restrictionam utilizarea metodei doar pt. imagini grayscale cu un canal (8 bit / pixel)
		if (channels_gray > 1)
			return;

		if (frameNum > 0) // daca nu este primul cadru
		{
			//------ SABLON DE PRELUCRARI PT. METODELE BACKGROUND SUBTRACTION -------
			// 
			// 2.5. Calcul imagine diferenta dintre cadrul current (gray) si fundal (backgnd)
			// Rezultatul se pune in matricea/imaginea diff
			// Backgnd este mereu primul cadru pentru al doilea cadru
			absdiff(gray, backgnd, diff);

			// 2.6. Modelarea pixelilor de fundal: Se actualizeaza matricea/imaginea model a fundalului (backgnd) conform celor 3 metode:
			// backgnd ul devine cadrul curent 
			backgnd = gray.clone();

			// 2.7. Binarizarea matricii diferenta (pt. toate metodele):
			// Se parcurge sistematic matricea diff
			// current pixel is forground (object) -> color in white
			for (int i = 0; i < diff.rows; i++)
				for (int j = 0; j < diff.cols; j++) {
					// daca valoarea pt. pixelul current 
					if (diff.at<uchar>(i, j) > Th)
						dst.at<uchar>(i, j) = 255; // current pixel is forground (object) -> color in white
						// selective running average -> no change to the background pixel model value
				}

			// 3. eroziunii + dilatari:
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			erode(dst, dst, element, Point(-1, -1), 1);
			dilate(dst, dst, element, Point(-1, -1), 1);

			//4. Se va masca imaginea diferenta finala (dst) de la pasul 3 cu ROI-ul fetei obtinut la pasul 1): 
			Mat temp = dst(faceROI);

			// 5. + 6.
			vector<mylist> candidates;
			candidates.clear();
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3); // matrice (3 canale) folositea pentru afisarea(color) a obiectelor detectate din regiunea de interes
			findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			Moments m;
			if (contours.size() > 0)
			{
				// iterate through all the top-level contours,
				// draw each connected component with its own random color
				int idx = 0;
				for (; idx >= 0; idx = hierarchy[idx][0])
				{
					const vector<Point>& c = contours[idx];
					m = moments(c); // calcul momente
					double arie = m.m00; // aria componentei conexe idx
					double xc = m.m10 / m.m00; // coordonata x a CM al componentei conexe idx
					double yc = m.m01 / m.m00; // coordonata y a CM al componentei conexe idx
					Scalar color(rand() & 255, rand() & 255, rand() & 255);
					drawContours(roi, contours, idx, color, CV_FILLED, 8, hierarchy);
					mylist elem;
					elem.arie = arie;
					elem.xc = xc;
					elem.yc = yc;
					candidates.push_back(elem);
				}
			}
			// 7. 
			if (candidates.size() < 2) {
				// closed:
				cout << "Nu avem 2 candidati"<<endl;
			}
			else {
				double arieL = 0.0, arieR = 0.0;
				mylist elemL;
				elemL.xc = 0.0;
				elemL.yc = 0.0;
				elemL.arie = 0.0;
				mylist elemR;
				elemR.xc = 0.0;
				elemR.yc = 0.0;
				elemR.arie = 0.0;
				for (mylist elem : candidates)
				{
					if (elem.xc < temp.cols / 2) { // jumatatea stanga
						if (elem.arie > arieL) {
							arieL = elem.arie;
							elemL = elem;
						}
					}
					else// jumatate dreapta
						if (elem.arie > arieR) {
							arieR = elem.arie;
							elemR = elem;
						}
				}
				double ky = 0.1;
				double kx1 = 0.3;
				double kx2 = 0.5;
				if (abs(arieR - arieL) < 110 && elemR.yc - elemL.yc < roi.rows * ky && kx1 * roi.cols<(elemR.xc - elemL.xc) && kx2 * roi.cols >(elemR.xc - elemL.xc)) {
					// open
					clipit = true;
					x++;
					DrawCross(roi, Point(elemL.xc, elemL.yc), 15, Scalar(255, 0, 0), 1);
					DrawCross(roi, Point(elemR.xc, elemR.yc), 10, Scalar(0, 0, 255), 1);
				}

				if (clipit) {
					cout << "CLIPIT: ";
					if (x % 2 == 0) {
						cout << "CLOSE" << endl;

						rectangle(src, faceROI, Scalar(0, 0, 255), 4, 8, 0);
					}
					else {
						cout << "OPEN" << endl;

						rectangle(src, faceROI, Scalar(0, 255, 0), 4, 8, 0);
					}
				}
			}
			imshow("roi", roi);
			//-------------------------------------------------------------------------
			// 3.9. Afiseaza imaginea sursa si destinatie
			imshow("sursa", frame); // show source
			imshow("result", src);
			imshow("dest", dst); // show destination
			// Plasati aici codul pt. vizualizarea oricaror rezultate intermediare
			// Ex: afisarea intr-o fereastra noua a imaginii diff
		}
		else // daca este primul cadru, modelul de fundal este chiar el
			backgnd = gray.clone();




		c = cvWaitKey(0); 
		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break; //ESC pressed
		}
	}
}

// laborator 11 sapt 10

CascadeClassifier fullbody_cascade; // cascade clasifier object for face
CascadeClassifier lowerbody_cascade; // cascade clasifier object for eyes
CascadeClassifier upperbody_cascade; // cascade clasifier object for eyes
//CascadeClassifier mcs_upperbody_cascade; // cascade clasifier object for eyes

void bodydetectanddisplay(const string& window_name, Mat frame)
{

	std::vector<Rect> bodies2;
	std::vector<Rect> bodies;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	fullbody_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(40, 150));

	for (int i = 0; i < bodies.size(); i++)
	{

		rectangle(frame, Point(bodies[i].x, bodies[i].y), Point(bodies[i].x + bodies[i].width, bodies[i].y + bodies[i].height), Scalar(255, 255, 0), 1, 8, 0);
		Point fullbodyCenter = RectCenter(bodies[i]);

		Mat bodiesROI = frame_gray(bodies[i]);

		int distance = 20;
		std::vector<Rect> upperbody;
		upperbody_cascade.detectMultiScale(frame_gray, upperbody, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(40, 75));
		std::vector<Rect> lowerbody;
		lowerbody_cascade.detectMultiScale(frame_gray, lowerbody, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(40, 90));
		double score;
		for (int j = 0; j < upperbody.size(); j++)
		{
			Point upperbodyCenter = RectCenter(upperbody[j]);
			rectangle(frame, upperbody[j], Scalar(255, 0, 255), 2, 8, 0);
			for (int l = 0; l < lowerbody.size(); l++)
			{
				Point lowerbodyCenter = RectCenter(lowerbody[l]);
				int arieLower = lowerbody[l].width * lowerbody[l].height;
				int arieUpper = upperbody[j].width * upperbody[j].height;
				int arieFull = bodies[i].width * bodies[i].height;
				//cout << upperbodyCenter.y <<" "<< lowerbodyCenter.y << endl;
				if (abs(upperbodyCenter.x - lowerbodyCenter.x) < distance && upperbodyCenter.y < lowerbodyCenter.y)
				{
					//printf("da1");
					if (abs(fullbodyCenter.x - upperbodyCenter.x) < distance && abs(fullbodyCenter.x - lowerbodyCenter.x) < distance
						&& RectArea(bodies[i] & (lowerbody[l] | upperbody[j])) > 0.7 * RectArea(bodies[i]))
						cout << 0.99 << endl;
					else
						if (upperbodyCenter.y - lowerbodyCenter.y < 150)
							cout << 0.66 << endl;
				}
				else
					if (abs(fullbodyCenter.x - upperbodyCenter.x) < distance && upperbodyCenter.y < fullbodyCenter.y
						&& RectArea(bodies[i] & upperbody[j]) > 0.5 * RectArea(upperbody[j]))
						cout << 0.66 << endl;
					else
					{
						if (abs(fullbodyCenter.x - lowerbodyCenter.x) < distance && lowerbodyCenter.y > fullbodyCenter.y
							&& RectArea(bodies[i] & lowerbody[l]) > 0.5 * RectArea(lowerbody[l]))
							cout << 0.66 << endl;
						else
							if ((abs(upperbodyCenter.y - lowerbodyCenter.y) > 140 || abs(fullbodyCenter.y - upperbodyCenter.y) > 140 || abs(fullbodyCenter.y - lowerbodyCenter.y) > 140)
								&& abs(upperbodyCenter.x - lowerbodyCenter.x) < distance && abs(fullbodyCenter.x - upperbodyCenter.x) < distance && abs(fullbodyCenter.x - lowerbodyCenter.x) < distance)
								cout << 0.33 << endl;
					}
			}
		}


		for (int j = 0; j < lowerbody.size(); j++)
		{
			rectangle(frame, lowerbody[j], Scalar(0, 255, 255), 2, 8, 0);
		}
	}
	imshow(window_name, frame);
}

//
void lab11()
{

	Mat src, dst;
	src = imread("Images/Persons/person_138.bmp", CV_LOAD_IMAGE_COLOR);
	dst = src.clone();

	String fullbody_cascade_name = "haarcascade_fullbody.xml";
	String upperbody_cascade_name = "haarcascade_upperbody.xml";
	String lowerbody_cascade_name = "haarcascade_lowerbody.xml";
	if (!fullbody_cascade.load(fullbody_cascade_name))
	{
		printf("Error loading fullbody cascades !\n");
		return;
	}
	if (!upperbody_cascade.load(upperbody_cascade_name))
	{
		printf("Error loading upperbody cascades !\n");
		return;
	}
	if (!lowerbody_cascade.load(lowerbody_cascade_name))
	{
		printf("Error loading lowerbody cascades !\n");
		return;
	}
	bodydetectanddisplay("WIN_DST", dst);
	waitKey(0);
}


// -----------------------------------PROIECT----------------------------

Mat processAndDisplayImages(const std::string& directory, int numSamples, Mat a, Mat b) {

	Mat samples[15]; //numSamples

	for (int i = 1; i <= numSamples; ++i) {
		std::string filename = directory +  std::to_string(i) + ".JPG";
		cout<< filename << endl;
		// Read the image
		samples[i] = imread(filename, IMREAD_COLOR);
		//imshow(to_string(i), samples[i]);
		//waitKey(0);
	}

	int height = a.rows;
	int width = a.cols;

	// 1. Split the sampels by the Lab color space 
	// Save the two chanels into 2 vectors (we don't need the lightness channel)
	Mat a_channel[15];
	Mat b_channel[15];
	for (int i = 1; i <= numSamples; i++) {
		std::vector<cv::Mat> channels;
		cv::split(samples[i], channels);
		cv::Mat L_channel = channels[0];
		a_channel[i] = channels[1];
		b_channel[i] = channels[2];
	}


	int histGlobabalaA[260] = { 0 };
	int histGlobabalaB[260] = { 0 };

	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				int value = a_channel[i].at<uchar>(k, l);
				histGlobabalaA[value] ++;
			}
		}
	}

	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				int value = b_channel[i].at<uchar>(k, l);
				histGlobabalaB[value]++;
			}
		}
	}

	double meanA = 0 ,meanB = 0;

	for (int i = 0; i <= 255; i++) {
		meanA += i*histGlobabalaA[i];
		meanB += i*histGlobabalaB[i];
	}
	
	meanA = (double) meanA / (numSamples * 16 * 16);
	meanB = (double) meanB / (numSamples * 16 * 16);
	printf("\nmeanA: %f             meanB: %f", meanA, meanB);

	// 2. compute the expection (miu) for both channels for all the samples' pixels:
	double ma = 0;
	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				ma += a_channel[i].at<uchar>(k, l);}}}
	ma = ma / (numSamples * 16 * 16);


	double mb = 0;
	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				mb += b_channel[i].at<uchar>(k, l);}}}
	mb = mb / (numSamples * 16 * 16);

	printf("\nma: %f           mb: %f\n", ma, mb);

	// 3. DEviatia standard:
	double devA = 0, devB = 0;
	for (int g = 0; g <= 255; g++) {
		devA += (g- meanA)*(g-meanA)*histGlobabalaA[g]/(numSamples * 16 * 16);
	}
	double devA2 = devA;
	devA = sqrt(devA);

	for (int g = 0; g <= 255; g++) {
		devB += (g - meanB) * (g - meanB) * histGlobabalaB[g] / (numSamples * 16 * 16);
	}
	double devB2 = devB;
	devB = sqrt(devB);

	printf("deviata A:%f           deviatia B:%f\n", devA2, devB2);

	
	Mat C = Mat(2, 2, CV_64FC1);

	double sum = 0;
	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				int x = a_channel[i].at<uchar>(k, l);
				sum += (x - ma) * (x - ma);
			}
		}
	}
	sum = sum / 16 / 16 / numSamples;
	C.at<double>(0, 0) = sum;

	//Covariance
	sum = 0;
	for (int i = 1; i <= numSamples; i++) {
		for (int k = 0; k < 16; k++) {
			for (int l = 0; l < 16; l++) {
				int x = b_channel[i].at<uchar>(k, l);
				sum += (x - mb) * (x - mb);
			}
		}
	}
	sum = sum / 16 / 16 / numSamples;
	C.at<double>(1, 1) = sum;
	C.at<double>(0, 1) = 0;
	C.at<double>(1, 0) = 0;

	printf("(0,0):%f            (1,1) %f", C.at<double>(0, 0), C.at<double>(1, 1));

	
	/*
	Mat Cinv = Mat(2, 2, CV_32FC1);
	float determinant = 1 / (C.at<float>(1, 1) * C.at<float>(0, 0) - C.at<float>(0, 1) * C.at<float>(1, 0));
	Cinv.at<float>(0, 0) = determinant * (C.at<float>(1, 1));
	Cinv.at<float>(1, 0) = -determinant * (C.at<float>(1, 0));
	Cinv.at<float>(0, 1) = -determinant * (C.at<float>(1, 0));
	Cinv.at<float>(1, 1) = determinant * (C.at<float>(0, 0));
	printf("\n%f %f %f %f", Cinv.at<float>(0, 0), Cinv.at<float>(0, 1), Cinv.at<float>(1, 0), Cinv.at<float>(1, 1));

	Mat Cinv = Mat(2, 2, CV_64FC1);
	Cinv.at<double>(0, 0) = 1/(C.at<double>(1, 1));
	Cinv.at<double>(1, 0) = 0;
	Cinv.at<double>(0, 1) = 0;
	Cinv.at<double>(1, 1) = 1/(C.at<double>(0, 0));
	*/

	Mat dst = a.clone();
	float max = -1000;
	float min = 300;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float valA = a.at<uchar>(i, j);
			float valB = b.at<uchar>(i, j);
			//printf(" %f, %f ", valA, valA);
			float mat[2];
			mat[0] = (valA - ma);
			mat[1] = (valB - mb);
			//printf(" %f, %f ", mat[0], mat[1]);
			float value = (mat[0] * mat[0] + mat[1] * mat[1]); // -> euler
		//	float value = (mat[0] * mat[0] / devB2 + mat[1] * mat[1] / devA2);
			//float value = (Cinv.at<double>(0,0)*mat[0]*mat[0] + Cinv.at<double>(1, 1) * mat[1] * mat[1] + 2 * Cinv.at<double>(0, 1) * mat[0] * mat[1] );
			//printf(" %f ", exponent);
			double result = sqrt(value);
			//printf(" %f ", result);
			dst.at<uchar>(i, j) = (uchar)result;
			if (max < result) max = result;
			if (min > result) min = result;

		}
	}

	// 0-1 -> 0 ->255 
	printf("\n max Value: %f        min Value : %f", max, min);
	cout << endl << "3.-------------------------------------------------------" << endl;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = (dst.at<uchar>(i, j)-min)/(max - min)*255 ;
		}
	}


	Mat dst2 = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = dst.at<uchar>(i, j);
			uchar neg = 255 - val;
			dst2.at<uchar>(i, j) = neg;
		}
	}

	return dst2;
}

Mat Binarizare(Mat src)
{
		int L = 255;
		Mat dst = Mat (src.rows,src.cols, CV_8UC1);

		int height = src.rows;
		int width = src.cols;

		int imax = 0, imin = 300;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int p = src.at<uchar>(i, j);
				if (imax < p) imax = p;
				if (imin > p) imin = p;
			}
		}
		float trashhold = (imax + imin) / 2;
		float lastT = 0;
		while ((trashhold - lastT) > 0.1) {
			//printf("%f ", trashhold - lastT);
			float medMin = 0, medMax = 0;
			int x = 0;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					uchar p = src.at<uchar>(i, j);
					if (p < trashhold) {
						medMin += p;
						x++;
					}
					else medMax += p;
				}
			}
			medMin = medMin / x;
			medMax = medMax / (height * width - x);
			lastT = trashhold;
			trashhold = (medMax + medMin) / 2;
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar p = src.at<uchar>(i, j);
				if (p < trashhold) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}

		return dst;
}

int** labels; // 0=> fundal // 1->label => complonenta
int label;

Mat LabelingBreadthFirstTraversal(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
	int height = src.rows;
	int width = src.cols;

	label = 0;
	labels = (int**)calloc((height + 1), sizeof(int*));  /* GLOBAL VARIABLE*/
	for (int i = 0; i < height + 1; i++) {
		labels[i] = (int*)calloc((width + 1), sizeof(int));
	}

	int di[8] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, 1, -1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar pixel = src.at<uchar>(i, j);
			if (pixel == 255 && labels[i][j] == 0) {		//pixel == white
				label++;
				std::queue<Point2i> Q;
				labels[i][j] = label;
				Q.push(Point2i(i, j));
				while (!Q.empty()) {
					Point2i q = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++) { // pt fiecare din cei 8 vecini
						Point n = Point(q.x + di[k], q.y + dj[k]);
						if (isInside(src, n.x, n.y) == true)
						{
							uchar neighbors = src.at<uchar>(n.x, n.y);
							if (neighbors == 255 && labels[n.x][n.y] == 0) { // color neighbor
								labels[n.x][n.y] = label;
								Q.push(n);
							}
						}
					}
				}
			}
		}
	}

	std::vector<Vec3b> colors;
	std::default_random_engine eng;
	std::uniform_int_distribution<int> d(0, 255);

	// cate o culoare pentru fiecare label:
	for (int i = 0; i <= label; i++)
		colors.push_back(Vec3b(d(eng), d(eng), d(eng))); 

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (labels[i][j] != 0)
				dst.at<Vec3b>(i, j) = colors[labels[i][j]];
		}
	}


	printf("\n componente After labeling: %d \n", label);
	cout << endl << "6.-------------------------------------------------------" << endl;
	//imshow("5. Etichetare: ", dst);
	return dst;
}

bool Label1Holes(Mat src, int label) {
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(src.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	Mat singleLevelHoles = Mat::zeros(src.size(), src.type());
	Mat multipleLevelHoles = Mat::zeros(src.size(), src.type());


	for (vector<Vec4i>::size_type idx = 0; idx < hierarchy.size(); ++idx)
	{
		if (hierarchy[idx][3] != -1)
			drawContours(singleLevelHoles, contours, idx, Scalar::all(255), CV_FILLED, 8, hierarchy);
	}

	bitwise_not(src, src);
	bitwise_and(src, singleLevelHoles, multipleLevelHoles);
	
	findContours(multipleLevelHoles, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	// Filter contours (remove the outer boundary)

	// Count the number of holes
	int numberOfHoles = contours.size();

	std::cout << "Number of holes: " << numberOfHoles <<" OF Label : "<< label<< std::endl;


	//Inverse source image.
	//imshow("Result0.jpg", src);

	//Holes before the bitwise AND operation.
	//imshow("Result1.jpg", singleLevelHoles);

	//Holes after the bitwise AND Operation.

	//imshow("Result2.jpg", multipleLevelHoles); // !!!!!!!!!!!!
	bool result = false;
	if (numberOfHoles >=1) result =  true; 

	return result;
}

Mat eulerEtichete(Mat src){
	bool* goodLabel = (bool*) calloc(label,sizeof(bool));
	// pentru reetichetare:
	int nrLabel = 0;
	int newLabel[1000];
	for (int i = 1; i <= label; i++) {
		Mat comp = src.clone();
		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				if (labels[x][y] != i) {
					comp.at<uchar>(x, y) = 0;
				}
			}
		}

		/*std::stringstream ss;
		ss << i;
		std::string stringValue = ss.str();
		imshow(stringValue, comp);*/

		goodLabel[i]= Label1Holes(comp, i);
		if (goodLabel[i])
		{
			nrLabel++;
			newLabel[i] = nrLabel;
		}
	}
	Mat dst = src.clone();
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			if (!goodLabel[labels[x][y]]) {
				dst.at<uchar>(x, y) = 0;  // eliminam componenta
				labels[x][y] = 0; // devine fundal
			}
			else {
				labels[x][y] = newLabel[labels[x][y]];
			}
		}
	}
	label = nrLabel;
	cout << "componente after Euler:" << label << endl;

	//Afisare noua etichetare:
	/*
	std::vector<Vec3b> colors;
	std::default_random_engine eng;
	std::uniform_int_distribution<int> d(0, 255);

	// cate o culoare pentru fiecare label:
	for (int i = 0; i <= label; i++)
		colors.push_back(Vec3b(d(eng), d(eng), d(eng)));

	Mat dst2 = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (labels[i][j] != 0)
				dst2.at<Vec3b>(i, j) = colors[labels[i][j]];
		}
	}
	imshow("5. Etichetare 2: ", dst2);*/


	
	imshow("After euler:", dst);
	return dst;

}

//-------------------Axe:

struct ElongationResults {
	double majorAxis;
	double minorAxis;
	double angle;  // unghi aza de alongatie
	int ri;    //rows pentru centrul de greutate
	int ci;	 // cols ----
};

ElongationResults calculateElongations(Mat binaryImage) {
	
	ElongationResults result;

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Find the minimum area bounding rectangle for the largest contour (assumed to be the object)
	cv::RotatedRect boundingBox = cv::minAreaRect(contours[0]);

	// Extract major and minor axes lengths
	double majorAxis = max(boundingBox.size.width, boundingBox.size.height);
	double minorAxis = min(boundingBox.size.width, boundingBox.size.height);

	cout << endl << "Major axis: " << majorAxis << "     Minor axis:  " << minorAxis << endl;
	cout << endl << "AR: " << majorAxis / minorAxis<<endl;
	float ar = majorAxis / minorAxis;
	result.majorAxis = majorAxis;
	result.minorAxis = minorAxis;
	return result;
}

ElongationResults Axa(Mat src) {
	int height = src.rows;
	int width = src.cols;

	// axe: 
	ElongationResults result = calculateElongations(src);

	Mat dst = src.clone();
	int ci = 0, ri = 0, surface = 0; //centre de masa
	for (int i = 0; i < height; i++) 
		for (int j = 0; j < width; j++) {
			if (dst.at<uchar>(i,j)==255) {  // pixel obiect
				ci = ci + j;
				ri = ri + i;
				surface++;
			}
		}
	ri = ri / surface;
	ci = ci / surface;
	result.ri = ri;
	result.ci = ci;

	// 3.2.3.Extragerea conturului mainii / fetei
	Labeling("Contur", dst, false);

	// 3.2.4. Desenarea axei de alungire a mainii /fetei
	float aux1 = 0, aux2 = 0, aux3 = 0, slope;
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			if (255 == dst.at<uchar>(i, j))
			{
				aux1 = aux1 + (i - ri) * (j - ci);
				aux2 = aux2 + (j - ci) * (j - ci);
				aux3 = aux3 + (i - ri) * (i - ri);
			}

	slope = atan2((2 * aux1), (aux2 - aux3)) / 2.0;
	result.angle = slope;
	
	// SHOW:
	printf("%f", slope);

	line(dst, Point(ci, ri), Point((int)(ci + 300 * cos(slope)), (int)(ri + 300 * sin(slope))), Vec3b(100, 100, 0));
	line(dst, Point(ci, ri), Point((int)(ci + 300 * cos(slope + PI)), (int)(ri + 300 * sin(slope + PI))), Vec3b(100, 100, 0));
	//imshow("Slope", dst);


	return result;
}

// -----------------TeamplateMatching:

int THRESHOLD_FACE = 70;

void templateMatching(Mat src, Mat originalImage, Mat color) {
	// pentru fiecare label:
	cout << endl << "7. 8. -------------------------------------------------------" << endl;
	std::string filename = "D:\\projects\\IOC_FaceDetect\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\sampleset\\average_face_clipped.jpg";

	Mat fin = src.clone();

	for (int l = 1; l <= label; l++) {
		cout << endl << "LABEL: " << l << endl;
		// A. verificam 7.  Calcul factor de alungire (aspect ratio - AR)
		Mat fata = src.clone();
		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				if (labels[x][y] != l) {
					fata.at<uchar>(x, y) = 0;
				}
			}
		}

		ElongationResults result = Axa(fata);
		double AR = result.majorAxis / result.minorAxis;
		if (1 <= AR && AR <= 3.5) {
			//PRINT:
			std::stringstream ss;
			ss << l;
			std::string stringValue = ss.str();
			imshow(stringValue, fata);
			cout << endl<< "Aspect ratio:" << AR << endl;

			
			
			// B. verificam 8. Template matching
			// 8.1 Deschidem template-ul
			Mat templateFace = imread(filename, IMREAD_GRAYSCALE);

			cout << "dsfdsf";
			// 8.2. Facem resize la template:
			cv::resize(templateFace, templateFace, cv::Size(result.minorAxis, result.majorAxis));
			//imshow("face template resize:", templateFace);
			cout << "dsfdsf";
			// 8.3. Facem rotire la template:
			double angle = -90 + abs(result.angle / PI) * 180;
			cout << endl << "angle in radians: " << result.angle << endl;
			cout << endl << "angle in degrees: " << angle << endl;

			Size size;
			size.height = templateFace.rows;
			size.width = templateFace.rows;
			Mat temp = Mat(size.height, size.width, CV_8UC1);

			for (int i = 0; i < size.height; i++)
				for (int j = 0; j < size.height; j++)
					temp.at<uchar>(i, j) = 0;

			for (int i = 0; i < size.height; i++) {
				int y = 0;
				for (int j = (size.height - templateFace.cols) / 2; j < (size.height + templateFace.cols) / 2; j++) {
					temp.at<uchar>(i, j) = templateFace.at<uchar>(i, y);
					y++;
				}
			}
			//angle +=10;
			cv::Point2f center(temp.cols / 2.0, temp.rows / 2.0);
			//imshow("strat", temp);
			cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
			cv::warpAffine(temp, temp, rotationMatrix, temp.size());
			//imshow("deleteme", temp);
			std::string stringValue2 = "Face rotation for label:" + ss.str();
			//line(temp, Point(temp.cols / 2, temp.cols / 2), Point((int)(temp.cols / 2 + 300 * cos(result.angle     )), (int)(temp.cols + 300 * sin(result.angle     ))), Vec3b(100, 100, 0));
			//line(temp, Point(temp.cols / 2, temp.cols / 2), Point((int)(temp.cols / 2 + 300 * cos(result.angle + PI)), (int)(temp.cols + 300 * sin(result.angle + PI))), Vec3b(100, 100, 0));
			imshow(stringValue2, temp);

			// 8.4. Suprapunem template-urile:
			// avem centrul fetelor in result si cunoastem centrul templateului:
			int c = size.height / 2;
			double medie = 0;
			int nr = 0;
			for (int i = 0; i < size.height; i++) {
				for (int j = 0; j < size.height; j++) {
					uchar pixelTemp = temp.at<uchar>(i, j);
					if (pixelTemp != 0 && isInside(originalImage, i + result.ri - c, j + result.ci - c)) { // pixelTEmp !=0 => parte din fata
						uchar pixelImg = originalImage.at<uchar>(i + result.ri - c, j + result.ci - c);
						originalImage.at<uchar>(i + result.ri - c, j + result.ci - c) = 255;
						double diference = abs(pixelTemp - pixelImg);
						medie += diference;
						nr++;
					}
				}
			}
			imshow("final", originalImage);
			medie = (double)medie / nr;
			printf("\n Diferenta intensitatilor: %f\n", medie);
			if (medie < THRESHOLD_FACE) {
				cout << "BUN!!";
				DrawCross(color, Point(result.ci, result.ri), 9, Scalar(0, 0, 255), 1);
			}
			else {
				cout << "RAU!!";
			}
		}
	}

	imshow("final Image color", color);
}


void test() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat image = imread(fname, IMREAD_COLOR);
		//imshow("input image", image);  // !!!!!!!!!!!!
		int height = image.rows;
		int width = image.cols;
		Mat imgLAB = Mat(height, width, CV_8UC3);
		// Post.  Aplicare FTJ gaussian pt. eliminare zgomote: essential sa il aplicati
		//GaussianBlur(image, image, Size(5, 5), 0, 0);
		
		// 1. Construirea unui model de culoare pentru piele: Lab + 2. Segmentarea zonelor
		cv::cvtColor(image, imgLAB, CV_BGR2Lab);
		// Split LAB image into L, a, and b channels
		std::vector<cv::Mat> channels;
		cv::split(imgLAB, channels);
		cv::Mat L_channel = channels[0];
		cv::Mat a_channel = channels[1];
		cv::Mat b_channel = channels[2];
		//imshow("A:", a_channel); imshow("B:", b_channel);
		// mmodifiy path for sampleset
		Mat likelihood = processAndDisplayImages("D:\\projects\\IOC_FaceDetect\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\sampleset\\", 13, a_channel, b_channel);
		 imshow("1+2. Likelihood Image:", likelihood); // !!!!!!!!!!!!

		// 3. Binarizare adaptiva
		Mat binarizare = Binarizare(likelihood);
		double threshold = 180;
		imshow("3. Binarizare adaptiva:", binarizare); // !!!!!!!!!!!!

		// 4. Operati morfologice
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

		erode(binarizare, binarizare, element, Point(-1, -1), 3);
		dilate(binarizare, binarizare, element, Point(-1, -1), 2);
		imshow("Dilate:", binarizare);
		imshow("4. Operati morfologice:", binarizare);

		// 5. Etichetare:
		Mat etich = LabelingBreadthFirstTraversal(binarizare);
		imshow("5. Etichetare: ", etich);

		// 6. Calcul nr. Euler
		binarizare = eulerEtichete(binarizare);

		// 7. Calcul factor de alungire (aspect ratio - AR) + 8. Template matching
		Mat gray;
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		templateMatching(binarizare, gray, image);


		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 111 - facial_detection \n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - show histogram (LAB1)\n");
		printf(" 11 - lab 3 \n");
		printf(" 12 - lab 4 \n");
		printf(" 13 - lab 5 \n");
		printf(" 14 - lab 5 cu henry \n");
		printf(" 15 - lab 5 video \n");
		printf(" 16 - lab 6  : method 1 2 3 \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				lab1();
				break;
			case 11:
				L3();
				break;
			case 12:
				MouseCallL4();
				break;
			case 13:
				findCorners();
				break;
			case 14:
				findCorners2();
				break;
			case 15: 
				findCornersVideo();
				break;
			case 16:
				BackgroundSubtraction();
				break;
			case 17:
				lab7HS();
				break;
			case 18:
				lab7LK();
				break;
			case 111:
				test();
				break;
			case 19:
				lab8();
				break;
			case 20:
				face_detection_mouth_nose();
				break;
			case 21:
				lab9_3();
				break;
			case 22:
				lab9_4();
				break;
			case 23:
				lab10();
				break;
			case 24:
				lab11();
				break;
		}
	}
	while (op!=0);
	return 0;
}