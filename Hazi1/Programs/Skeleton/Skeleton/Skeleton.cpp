//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Rittgasszer Ákos
// Neptun : Z8WK8D
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";



GPUProgram gpuProgram; // vertex and fragment shaders

bool operator==(vec2 A, vec2 B)
{
	if (fabs((A - B).x) < 0.0001 && fabs((A - B).y) < 0.0001)
		return true;
	return false;
}
bool linesCross(vec2 A1, vec2 A2, vec2 B1, vec2 B2)
{
	float a = (B1.x - B2.x) / (A1.x - A2.x);
	float b = (B2.x - A2.x) / (A1.x - A2.x);

	float t2 = (B2.y - A1.y * b - A2.y * (1 - b)) / (a * (A1.y - A2.y) - B1.y + B2.y);
	float t1 = a * t2 + b;

	if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1))
		return true;
	return false;
}
float getAngle(vec2 P)
{
	float angle;

	float a1 = asinf(P.y);
	float a2 = M_PI - asinf(P.y);
	float a3 = acosf(P.x);
	float a4 = 2 * M_PI - acosf(P.x);

	if (a1 < 0)
		a1 += 2 * M_PI;
	if (a2 < 0)
		a2 += 2 * M_PI;
	if (a3 < 0)
		a3 += 2 * M_PI;
	if (a4 < 0)
		a4 += 2 * M_PI;



	if (a1 < a3 + 0.01 && a1 >= a3 - 0.01)
		angle = a1;
	else if (a1 < a4 + 0.01 && a1 >= a4 - 0.01)
		angle = a1;
	else if (a2 < a3 + 0.01 && a2 >= a3 - 0.01)
		angle = a3;
	else if (a2 < a4 + 0.01 && a2 >= a4 - 0.01)
		angle = a2;

	return angle;
}



class Triangle
{
	unsigned int vao;
	vec2 A_;
	vec2 B_;
	vec2 C_;
	vec2 points[3];
	vec3 color_;

	mat4 M()
	{
		return mat4(1, 0, 0, 0,    // MVP matrix, 
					0, 1, 0, 0,    // row-major!
					0, 0, 1, 0,
					0, 0, 0, 1);
	}
public:
	Triangle(vec2 A, vec2 B, vec2 C, vec3 color) : A_{ A }, B_{ B }, C_{ C }, color_{color}
	{
		points[0] = A;
		points[1] = B;
		points[2] = C;
	}


	void create()
	{
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)


		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * 3,  // # bytes
			points,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void draw()
	{
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color_.x, color_.y, color_.z); // 3 floats

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, M());	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 3 /*# Elements*/);
	}
};

class SiriusTriangle
{
	static const int circlePointNum = 20;
	

	unsigned int vao;
	int numOfPoints;
	vec2 trianglePoints[3];
	std::vector<vec3> siriusPoints;
	vec2 vertices[(circlePointNum) * 3];
	vec3 fillColor_;
	vec3 lineColor_;


	mat4 M(float r, float x, float y)
	{
		return mat4(r, 0, 0, 0,    // MVP matrix, 
			0, r, 0, 0,    // row-major!
			0, 0, 0, 0,
			x, y, 0, 1);
	}
	float size(vec2 A, vec2 B)
	{
		float size;

		size = (length(A - B)) / (1 - A.x*A.x - A.y*A.y);

		return size;
	}

	void calculateTriangle()
	{
		std::vector<vec2> side1 = calculateSide(trianglePoints[0], trianglePoints[1]);
		std::vector<vec2> side2 = calculateSide(trianglePoints[1], trianglePoints[2]);
		std::vector<vec2> side3 = calculateSide(trianglePoints[2], trianglePoints[0]);


		int idx = 0;

		for (size_t i = 0; i < side1.size(); ++i)
		{
			vertices[idx] = side1[i];
			++idx;
		}
		for (size_t i = 0; i < side2.size(); ++i)
		{
			vertices[idx] = side2[i];
			++idx;
		}
		for (size_t i = side3.size(); i > 0; --i)
		{
			vertices[idx] = side3[side3.size() - i];
			++idx;
		}

	}
	std::vector<vec2> calculateSide(vec2 A, vec2 B)
	{
		float r, x, y;

		vec2 F((A.x + B.x) / 2, (A.y + B.y) / 2);					//AB felezopontja
		vec2 ABvec = A - B;											//(u; v) AB vektor


		if (B.y - 0.001 < A.y && A.y < B.y + 0.001) //Ha A ket ponton atmeno egyenes parhuzamos az y tengellyel
		{
			x = F.x;
		}
		else
		{
			//Pitagorasz tetel, a kor kozeppontja rajta van aket pont alkotta szakasz felezomerolegesen
			float u = ABvec.x;
			float v = ABvec.y;


			float f = v * (A.x * A.x + A.y * A.y + 1);
			float g = 2 * A.y * (u * F.x + v * F.y);
			float h = 2 * (v * A.x - u * A.y);

			x = (f - g) / h;
		}


		y = (A.x * A.x + A.y * A.y - 2.0f * A.x * x + 1.0f) / (2.0f * A.y);
		r = sqrtf((A.x - x) * (A.x - x) + (A.y - y) * (A.y - y));

		siriusPoints.push_back(vec3(x, y, r));

		std::vector<vec2> side;


		//Pontok transzformalasa egysegkorre
		vec2 A_ = (A - vec2(x, y)) / r;
		vec2 B_ = (B - vec2(x, y)) / r;


		//A szogek kiszamitasa, annak figyelembevetlevel, hogy egy koordinatahoz ket szog tartozik
		float angleA = getAngle(A_);
		float angleB = getAngle(B_);


		float fi;

		//A 2 pont kozotti iv mindig kisebb mint egy felkoriv, 
		//r^2 + 1 = x^2 + y^2, ahol r^2 > 0 ==> 1 < r^2 + 1 ami azt jelenti, 
		//hogy (x; y) egynel nagyobb tavolsagra van az origotol
		if (fabs(angleB - angleA) > M_PI)
			fi = (angleB + 2 * M_PI - angleA) / circlePointNum;
		else
			fi = (angleB - angleA) / circlePointNum;


		if (fi < 0)
		{
			for (float angle = angleA; !(angle > angleB - 0.00001 && angle < angleB + 0.00001);)
			{
				if (angle >= 2 * M_PI)
					angle -= 2 * M_PI;

				float x_, y_;

				y_ = sinf(angle) * r + y;
				x_ = cosf(angle) * r + x;

				side.push_back(vec2(x_, y_));

				angle += fi;
				if (angle >= 2 * M_PI)
					angle -= 2 * M_PI;
			}
		}
		else
		{
			for (float angle = angleA; !(angle > angleB - 0.00001 && angle < angleB + 0.00001); )
			{
				if (angle >= 2 * M_PI)
					angle -= 2 * M_PI;


				float x_, y_;

				y_ = sinf(angle) * r + y;
				x_ = cosf(angle) * r + x;

				side.push_back(vec2(x_, y_));

				angle += fi;
				if (angle >= 2 * M_PI)
					angle -= 2 * M_PI;
			}
		}

		return side;
	}
	
	void calculateTriangleData()
	{
		float alpha, beta, gamma;

		alpha = calculateAngles(siriusPoints[0], siriusPoints[1]) * 180 / M_PI;
		beta = calculateAngles(siriusPoints[1], siriusPoints[2]) * 180 / M_PI;
		gamma = calculateAngles(siriusPoints[2], siriusPoints[0]) * 180 / M_PI;

		float sum = alpha + beta + gamma;

		printf("alpha: %f  beta: %f  gamma: %f  summa: %f\n", alpha, beta, gamma, sum);

		float a, b, c;

		a = calculateSideLength(0);
		b = calculateSideLength(1);
		c = calculateSideLength(2);

		printf("a: %f  b: %f  c: %f \n", a, b, c);
	}

	float calculateAngles(vec3 c1, vec3 c2)
	{
		float angle;
		vec2 P;

		for (size_t idx = 0; idx < 3; ++idx)
		{
			if (fabs(length(trianglePoints[idx] - vec2(c1.x, c1.y)) - c1.z) < 0.001 &&
				fabs(length(trianglePoints[idx] - vec2(c2.x, c2.y)) - c2.z) < 0.001)
			{
				P = trianglePoints[idx];
				break;
			}
		}

		vec2 v1 = P - vec2(c1.x, c1.y);
		vec2 v2 = P - vec2(c2.x, c2.y);

		//skalaris szrzat
		angle = acosf((dot(v1, v2) / (length(v1) * length(v2))));
		
		if (angle > M_PI)
			angle -= 2 * M_PI;
		if (angle < 0)
			angle *= -1;


		if (angle > M_PI / 2)
			angle = M_PI - angle;

		return angle;
	}
	float calculateSideLength(int idx)
	{
		float length = 0;

		for (int i = idx * circlePointNum; i < (idx + 1) * circlePointNum; ++i)
		{
			if(i == 3 * circlePointNum - 1)
				length += size(vertices[i], vertices[0]);
			else
				length += size(vertices[i], vertices[i + 1]);
		}

		return length;
	}

	std::vector<Triangle> Triangles()
	{
		std::vector<Triangle> triangles;
		std::vector<vec2> points;
		for (int idx = 0; idx < circlePointNum * 3; ++idx)
			points.push_back(vertices[idx]);

		//fulvago algoritmus
		for (size_t idx = 0; idx < points.size(); )
		{
			if (3 == points.size())
			{
				triangles.push_back(Triangle(points[0], points[1], points[2], fillColor_));
				break;
			}

			size_t idxBefore, idxAfter;


			if (idx == 0)
				idxBefore = points.size() - 1;
			else
				idxBefore = idx - 1;
			if (idx == points.size() - 1)
				idxAfter = 0;
			else
				idxAfter = idx + 1;

			vec2 before = points[idxBefore];
			vec2 after = points[idxAfter];

			if (ear(idxBefore, idxAfter, points))
			{
				triangles.push_back(Triangle(before, points[idx], after, fillColor_));
				points.erase(points.begin() + idx);
				idx = 0;
			}
			else
			{
				++idx;
			}
		}
		return triangles;
	}
	bool ear(int idxBefore, int idxAfter, std::vector<vec2> points)
	{
		vec2 before = points[idxBefore];
		vec2 after = points[idxAfter];

		//Metsz-e masik oldalt?
		//
		for (int i = 0; i < circlePointNum * 3; ++i)
		{

			int bIdx;
			if (i == circlePointNum * 3 - 1)
				bIdx = 0;
			else
				bIdx = i + 1;


			vec2 a = vertices[i];
			vec2 b = vertices[bIdx];

			if (a == after || b == after || a == before || b == before)
				continue;

			if (linesCross(a, b, before, after))
				return false;

		}


		//kulso vagy belso hur
		//
		vec2 F = (before + after) / 2;
		vec2 I(10.0f, 10.0f);
		int crosses = 0;

		for (int idx = 0; idx < circlePointNum * 3; ++idx)
		{
			int bIdx;
			if (idx == (circlePointNum * 3 - 1))
				bIdx = 0;
			else
				bIdx = idx + 1;

			vec2 a = vertices[idx];
			vec2 b = vertices[bIdx];			

			if (linesCross(a, b, F, I))
				++crosses;
		}

		if (crosses % 2 == 1)
			return true;
		return false;
	}

	void fill()
	{
		std::vector<Triangle> triangles = Triangles();

		for (size_t idx = 0; idx < triangles.size(); ++idx)
		{
			triangles[idx].create();
			triangles[idx].draw();
		}
	}
	void drawLines()
	{
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, lineColor_.x, lineColor_.y, lineColor_.z); // 3 floats

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, M(1, 0, 0));	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, (circlePointNum) * 3 /*# Elements*/);
	}

public:

	SiriusTriangle(vec3 lineColor, vec3 fillColor) 
	{
		lineColor_ = lineColor;
		fillColor_ = fillColor;
		numOfPoints = 0; 
	}

	void AddPoint(float x, float y)
	{
		trianglePoints[numOfPoints] = vec2(x, y);
		++numOfPoints;
	}

	void create()
	{
		calculateTriangle();
		calculateTriangleData();

		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)


		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * (circlePointNum + 1) * 3,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}
	void draw()
	{
		if (numOfPoints == 3)
		{
			fill();
			drawLines();		
		}
	}
};

class Circle
{
	unsigned int vao;	   // virtual world on the GPU
	static const int circlePointNum = 100;
	float x_;
	float y_;
	float r_;
	vec3 color_;

	mat4 M(float x, float y, float r)
	{
		return mat4(r, 0, 0, 0,    // MVP matrix, 
					0, r, 0, 0,    // row-major!
					0, 0, 1, 0,
					x, y, 0, 1);
	}

public:
	Circle(float x, float y, float r, vec3 color) : x_{ x }, y_{ y }, r_{ r }, color_{color} {}

	void create()
	{
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

		vec2 vertices[circlePointNum];
		for (int idx = 0; idx < circlePointNum; ++idx)
		{
			float fi = idx * 2 * M_PI / circlePointNum;
			vertices[idx] = vec2(cosf(fi), sinf(fi));
		}


		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * circlePointNum,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void draw()
	{

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color_.x, color_.y, color_.z); // 3 floats

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, M(x_, y_, r_));	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, circlePointNum /*# Elements*/);
	}
};

Circle basicCircle(0.0, 0.0, 1, vec3(0.6, 0.6, 0.6));
SiriusTriangle siriusTriangle(vec3(0, 1, 0), vec3(0, 0, 1));
int clicks = 0;



// Initialization, create an OpenGL context
void onInitialization() 
{
	glViewport(0, 0, windowWidth, windowHeight);

	basicCircle.create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() 
{
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	
	basicCircle.draw();
	siriusTriangle.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) 
{
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
} 

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) 
{
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY)		// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
{	
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY)			 // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
{
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
	{
		++clicks;
		siriusTriangle.AddPoint(cX, cY);
		if (clicks == 3)
		{
			clicks = 0;
			siriusTriangle.create();
		}
	}

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() 
{
}
