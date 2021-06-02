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
// Nev    : Rittgasszer Akos
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
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"
#include <iostream>



float rnd() { return (float)rand() / (float)RAND_MAX; }

const float epsilon = 0.0001f;

vec3 operator/(vec3 v1, vec3 v2)
{
	return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

enum MaterialType{ROUGHT, REFLECTIVE};

struct Material 
{
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	Material(MaterialType type) : type_(type) {}
	MaterialType type_;
};

struct RoughtMaterial : public Material
{
	RoughtMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGHT)
	{
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

struct ReflectiveMaterial : public Material
{
	ReflectiveMaterial(vec3 n, vec3 k) : Material(REFLECTIVE)
	{
		F0 = ((n - vec3(1, 1, 1)) * (n - vec3(1, 1, 1)) + k * k) / 
			 ((n + vec3(1, 1, 1)) * (n - vec3(1, 1, 1)) + k * k);
	}
};

struct Hit 
{
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray 
{
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
	vec3 reflect(vec3 normal)
	{
		return dir - normal * dot(normal, dir) * 2.0f;
	}
	vec3 Fresnel(vec3 normal, Material* m)
	{
		float cos = -dot(dir, normal);
		return m->F0 + (vec3(1, 1, 1) - m->F0) * pow(1 - cos, 5);
	}
};

class Intersectable 
{
public:
	virtual Hit intersect(const Ray& ray) = 0;
protected:
	Material* material_;
};

class Sphere : public Intersectable
{
public:
	Sphere(const vec3& center, float r, Material* material) 
	{
		center_ = center;
		radius_ = r;
		material_ = material;
	}

	Hit intersect(const Ray& ray) 
	{
		Hit hit;

		vec3 dist = ray.start - center_;

		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius_ * radius_;

		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center_) * (1.0f / radius_);
		hit.material = material_;

		return hit;
	}
	
private:
	vec3 center_;
	float radius_;
};

class Ellipsoid : public Intersectable
{
public:
	Ellipsoid(float a, float b, float c, vec3 center, Material* material)
	{
		a_ = a;
		b_ = b;
		c_ = c;
		center_ = center;
		material_ = material;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		float a = powf(ray.dir.x / a_, 2.0f) + powf(ray.dir.y / b_, 2.0f) + powf(ray.dir.z / c_, 2.0f);
		float b = (2 * (ray.start.x - center_.x) * ray.dir.x) / (a_ * a_) + 
				  (2 * (ray.start.y - center_.y) * ray.dir.y) / (b_ * b_) +
				  (2 * (ray.start.z - center_.z) * ray.dir.z) / (c_ * c_);
		float c = powf((ray.start.x - center_.x) / a_, 2.0f) +
				  powf((ray.start.y - center_.y) / b_, 2.0f) +
				  powf((ray.start.z - center_.z) / c_, 2.0f) - 1;


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal = gradf(hit.position);
		hit.material = material_;


		return hit;
	}

	vec3 gradf(vec3 point)
	{
		vec3 v(2 * (point.x - center_.x) / (a_ * a_),
			   2 * (point.y - center_.y) / (b_ * b_),
			   2 * (point.z - center_.z) / (c_ * c_));
		return v / length(v);
	}

	float f(vec3 point)
	{
		return 0;
	}

protected:
	float a_, b_, c_;
	vec3 center_;
};

class EllipsoidWithHole : public Ellipsoid
{
public:
	EllipsoidWithHole(float a, float b, float c, vec3 center, Material* material, float h = 0.95, vec3 d = vec3(0, 1, 0), vec3 o = vec3(0, 0, 0))
		: Ellipsoid(a, b, c, center, material)
	{
		this->height_ = h;
		this->O = o;
		this->dir = dir;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		float a = powf(ray.dir.x / a_, 2.0f) + powf(ray.dir.y / b_, 2.0f) + powf(ray.dir.z / c_, 2.0f);
		float b = (2 * (ray.start.x - center_.x) * ray.dir.x) / (a_ * a_) +
			(2 * (ray.start.y - center_.y) * ray.dir.y) / (b_ * b_) +
			(2 * (ray.start.z - center_.z) * ray.dir.z) / (c_ * c_);
		float c = powf((ray.start.x - center_.x) / a_, 2.0f) +
			powf((ray.start.y - center_.y) / b_, 2.0f) +
			powf((ray.start.z - center_.z) / c_, 2.0f) - 1;


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		if(hit.position.y > height_)
		{
			Hit h;
			return h;
		}

		hit.normal = gradf(hit.position);
		hit.material = material_;


		return hit;
	}

private:
	float r_;
	float height_;
	vec3 O;
	vec3 dir;
};

class Hiperboloid : public Intersectable
{
public:
	Hiperboloid(float a, float b, float c, float height, vec3 center, Material* material)
	{
		a_ = a;
		b_ = b;
		c_ = c;
		height_ = height;
		center_ = center;
		material_ = material;
		/*Q_ = {1.0f/(a *a),   0,   0,  -center.x/(a*a),
				0, -1.0f/(b * b),   0,  center.y / (b * b),
				0,   0, 1.0f/(c * c),  -center.z / (c * c),
				-center.x / (a * a),   center.y / (b * b),   -center.z / (c * c), -1.0f + center.x*center.x - center.y * center.y + center.z * center.z
		};*/

	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		float a = powf(ray.dir.x / a_, 2.0f) - powf(ray.dir.y / b_, 2.0f) + powf(ray.dir.z / c_, 2.0f);
		float b = (2 * (ray.start.x - center_.x) * ray.dir.x) / (a_ * a_) -
			(2 * (ray.start.y - center_.y) * ray.dir.y) / (b_ * b_) +
			(2 * (ray.start.z - center_.z) * ray.dir.z) / (c_ * c_);
		float c = powf((ray.start.x - center_.x) / a_, 2.0f) -
			powf((ray.start.y - center_.y) / b_, 2.0f) +
			powf((ray.start.z - center_.z) / c_, 2.0f) - 1;


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;


		if (hit.position.y > center_.y + height_ / 2 || hit.position.y < center_.y - height_ / 2)
		{
			if(fabs(hit.t - t1) < epsilon)
				hit.t = t2;
			else
				hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			if (hit.position.y > center_.y + height_ / 2 || hit.position.y < center_.y - height_ / 2)
			{
				Hit h;
				return h;
			}
		}

		hit.normal = gradf(hit.position);
		hit.material = material_;


		return hit;
	}

	vec3 gradf(vec3 point)
	{
		vec3 v(2 * (point.x - center_.x) / (a_ * a_),
			-2 * (point.y - center_.y) / (b_ * b_),
			2 * (point.z - center_.z) / (c_ * c_));
		return v / length(v);
	}

	float f(vec3 point)
	{
		return 0;
	}

protected:
	float a_, b_, c_;
	float height_;
	vec3 center_;
};

class HalfHiperboloid :public Hiperboloid
{
public:
	HalfHiperboloid(float a, float b, float c, float height, vec3 center, Material* material): Hiperboloid(a, b, c, height, center, material)
	{

	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		float a = powf(ray.dir.x / a_, 2.0f) - powf(ray.dir.y / b_, 2.0f) + powf(ray.dir.z / c_, 2.0f);
		float b = (2 * (ray.start.x - center_.x) * ray.dir.x) / (a_ * a_) -
			(2 * (ray.start.y - center_.y) * ray.dir.y) / (b_ * b_) +
			(2 * (ray.start.z - center_.z) * ray.dir.z) / (c_ * c_);
		float c = powf((ray.start.x - center_.x) / a_, 2.0f) -
			powf((ray.start.y - center_.y) / b_, 2.0f) +
			powf((ray.start.z - center_.z) / c_, 2.0f) - 1;


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;


		if (hit.position.y > center_.y + height_ || hit.position.y < center_.y )
		{
			if (fabs(hit.t - t1) < epsilon)
				hit.t = t2;
			else
				hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			if (hit.position.y > center_.y + height_ || hit.position.y < center_.y)
			{
				Hit h;
				return h;
			}
		}

		hit.normal = gradf(hit.position);
		hit.material = material_;


		return hit;
	}
};

class Paraboloid : public Intersectable
{
public:
	Paraboloid(float a, float b, float h, vec3 center, Material* material)
	{
		a_ = a;
		b_ = b;
		center_ = center;
		height_ = h;
		material_ = material;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		float a = powf(ray.dir.x / a_, 2.0f) + powf(ray.dir.z / b_, 2.0f);
		float b = (2 * (ray.start.x - center_.x) * ray.dir.x) / (a_ * a_) +
			ray.dir.y +
			(2 * (ray.start.z - center_.z) * ray.dir.z) / (b_ * b_);
		float c = powf((ray.start.x - center_.x) / a_, 2.0f) +
			ray.start.y - center_.y +
			powf((ray.start.z - center_.z) / b_, 2.0f) - 1;


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;


		if (hit.position.y > center_.y + height_ / 2 || hit.position.y < center_.y - height_ / 2)
		{
			if (fabs(hit.t - t1) < epsilon)
				hit.t = t2;
			else
				hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			if (hit.position.y > center_.y + height_ / 2 || hit.position.y < center_.y - height_ / 2)
			{
				Hit h;
				return h;
			}
		}

		hit.normal = gradf(hit.position);
		hit.material = material_;


		return hit;
	}

	vec3 gradf(vec3 point)
	{
		vec3 v(2 * (point.x - center_.x) / (a_ * a_),
			1,
			2 * (point.z - center_.z) / (b_ * b_));
		return v / length(v);
	}

	float f(vec3 point)
	{
		return 0;
	}

private:
	float a_, b_;
	float height_;
	vec3 center_;
};

class Camera 
{
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) 
	{
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) 
	{
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light 
{
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) 
	{
		direction = normalize(_direction);
		Le = _Le;
	}
};


class Scene 
{
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<vec3> samplePoints;
	Camera camera;
	vec3 La;
public:
	void build() 
	{
		samplePoints = getSamplePoints(50);


		vec3 eye = vec3(0, 0, 2.9), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(10, -10, 10), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.2f, 0.3f, 0.3f), ks(20, 20, 20);
		Material* material = new RoughtMaterial(kd, ks, 50);

		objects.push_back(new EllipsoidWithHole(3, 1, 3, vec3(0, 0, 0), material, 0.98));

		material = new RoughtMaterial(vec3(0.2f, 0.6f, 0.2f), ks, 50);
		objects.push_back(new Ellipsoid(0.2, 0.4, 0.3, vec3(-1, 0.1, -1), material));

		material = new RoughtMaterial(vec3(0.6f, 0.2f, 0.2f), ks, 50);
		objects.push_back(new Hiperboloid(0.2, 0.4, 0.3, 0.5, vec3(1, -0.5, -1), material));
		
		Material *gold = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		objects.push_back(new Paraboloid(0.4, 0.4, 2, vec3(0, -0.5, -2), gold));

		Material * silver = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		objects.push_back(new HalfHiperboloid(0.596992f, 0.596992f, 0.596992f, 0.6, vec3(0, 0.98f, 0), silver));
	}

	void render(std::vector<vec4>& image) 
	{
		for (int Y = 0; Y < windowHeight; Y++) 
		{
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) 
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) 
	{
		Hit bestHit;
		for (Intersectable* object : objects) 
		{
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) 
	{	
		for (Intersectable* object : objects) 
			if (object->intersect(ray).t > 0 && object->intersect(ray).material->type_ == ROUGHT)
				return true;
		return false;
	}

	std::vector<vec3> getSamplePoints(int n)
	{
		float r = 0.596992f;
		float h = 0.98f;
		std::vector<vec3> points;

		for (int idx = 0; idx < n; ++idx)
		{
			float x = rnd() * 2 * r - r;
			float z = rnd() * 2 * r - r;
			if (sqrtf(x * x + z * z) <= r * r)
				points.push_back(vec3(x, h, z));
		}
		return points;
	}

	float A = 0.596992f * 0.596992f * M_PI;
	vec3 trace(Ray ray, int depth = 0)
	{
		if (depth > 5)
			return La;

		Hit hit = firstIntersect(ray);

		if (hit.t < 0)
			return La + pow(dot(ray.dir, lights[0]->direction), 10) * lights[0]->Le;
			

		vec3 outRadiance = vec3(0, 0, 0);

		if (hit.material->type_ == REFLECTIVE)
		{
			Ray newRay(hit.position + hit.normal * epsilon, ray.reflect(hit.normal));
			outRadiance = outRadiance + trace(newRay, depth + 1) * ray.Fresnel(hit.normal, hit.material);
		}
		if (hit.material->type_ == ROUGHT)
		{
			outRadiance = hit.material->ka * La;

			vec3 start = hit.position + hit.normal * epsilon;
			for (vec3 point : samplePoints) 
			{
				Ray shadowRay(start, point - start);
				float cosTheta = dot(hit.normal, shadowRay.dir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay))
				{
					float r = length(point - start);
					float omega = (A / samplePoints.size()) * dot(vec3(0, 1, 0), shadowRay.dir) / (r * r);
					
					outRadiance = outRadiance + trace(shadowRay, depth + 1) * cosTheta * hit.material->kd * omega;
					vec3 halfway = normalize(-ray.dir + shadowRay.dir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) 
						outRadiance = outRadiance + trace(shadowRay, depth + 1) * hit.material->ks * powf(cosDelta, hit.material->shininess) * omega;
				}
			}
		}
	
		return outRadiance;
		
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


class FullScreenTexturedQuad 
{
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() 
	{
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() 
{
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() 
{
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) 
{

}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) 
{

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) 
{

}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) 
{

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() 
{

}