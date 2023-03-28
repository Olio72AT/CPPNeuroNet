#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;

#define Train_Set_Size 20
#define PI 3.141592653589793238463
#define N 5
#define epsilon 0.05
#define epoch 100000
#define width 800
#define height 800
#define multiply 100

double c[N] = {};
double W[N] = {};
double V[N] = {};
double b = 0;


double sigmoid(double x) {
	return (1.0f / (1.0f + std::exp(-x)));
}

double f_theta(double x) {
	double result = b;
	for (int i = 0; i < N; i++) {
		result += V[i] * sigmoid(c[i] + W[i] * x);
	}
	return result;
}

void train(double x, double y) {
	for (int i = 0; i < N; i++) {
		W[i] = W[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * x * 
               (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
	for (int i = 0; i < N; i++) {
		V[i] = V[i] - epsilon * 2 * (f_theta(x) - y) * sigmoid(c[i] + W[i] * x);
	}
	b = b - epsilon * 2 * (f_theta(x) - y);
	for (int i = 0; i < N; i++) {
		c[i] = c[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * 
               (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
}

int main() {
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		W[i] = 2 * rand() / RAND_MAX -1;
		V[i] = 2 * rand() / RAND_MAX -1;
		c[i] = 2 * rand() / RAND_MAX -1;
	}
	vector<pair<double, double>> trainSet;
	trainSet.resize(Train_Set_Size);

	for (int i = 0; i < Train_Set_Size; i++) {
		trainSet[i] = make_pair(i * 2 * PI / Train_Set_Size, sin(i * 2 * PI / Train_Set_Size));
	}

	for (int j = 0; j < epoch; j++) {
		for (int i = 0; i < Train_Set_Size; i++) {
			train(trainSet[i].first, trainSet[i].second);
		}
		std::cout << j << "\r";
	}

	//Plot the results
	vector<float> x;
	vector<float> y1, y2;

	for (int i = 0; i < 1000; i++) {
		x.push_back(i * 2 * PI / 1000);
		y1.push_back(sin(i * 2 * PI / 1000));
		y2.push_back(f_theta(i * 2 * PI / 1000));
	}

    std::ofstream outfile;
    outfile.open("index3.html");
    
    
	outfile << "<!DOCTYPE html>\n<html>\n<head>\n" \
		"<title>Canvas 2D Plot</title>\n</head>\n<body>\n";
	
	string temp = "<p>Sample for neuronal net for learning a sinus wave in c++ <br> LG Oliver <br></p>\n" \
		"<canvas id=\"myCanvas\" width=\"1000\" height=\"1000\"></canvas>";
	outfile << temp;
	temp = 
		"<script>\n// Get the canvas element\nvar canvas = document.getElementById(\"myCanvas\");" \
		"// Get the canvas context\n" \
		"var ctx = canvas.getContext(\"2d\");\n" \
		"var data = [\n";
	outfile << temp;
	

    for (int k = 0; k < x.size(); k++) {
		outfile << "{x: "<< x[k] * multiply + (width/4) << ", y: " << y1[k] * multiply + (height/2) << "},\n";
	}

	for (int k = 0; k < x.size()-1; k++) {
		outfile << "{x: "<< x[k] * multiply + (width/4) << ", y: " << y2[k] * multiply + (height/2) << "},\n";
	}

	outfile << "{x: "<< x[(x.size()-1)] * multiply + (width/4) << ", y: " << y2[(x.size()-1)] * multiply + (height/2) << "}\n ];\n";

	outfile << 
		"// Set the line style\n" \
		"ctx.lineWidth = 2;\n" \
		"ctx.strokeStyle = '#333';\n" \

		"\n\n// Plot the data points" \
		"\n ctx.beginPath();" \
		"\n ctx.moveTo(data[0].x, data[0].y);" \
		"\n for (var i = 1; i < data.length; i++) {	ctx.lineTo(data[i].x, data[i].y);} " \
		"\n ctx.stroke();" \
	    "\n </script></body></html>";


    // Close the file
    outfile.close();

	return 0;
}