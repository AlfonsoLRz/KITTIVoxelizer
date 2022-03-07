#include "stdafx.h"
#include "PointCloud.h"

#include <filesystem>
#include "Graphics/Application/TextureList.h"
#include "Graphics/Core/ShaderList.h"
#include "Graphics/Core/VAO.h"

// Initialization of static attributes
const std::string	PointCloud::WRITE_POINT_CLOUD_FOLDER = "PointClouds/";

/// Public methods

PointCloud::PointCloud(const std::string& filename, const bool useBinary, const mat4& modelMatrix) :
	Model3D(modelMatrix, 1), _filename(filename), _useBinary(useBinary), _maxLabel(0)
{
}

PointCloud::~PointCloud()
{
}

bool PointCloud::load(const mat4& modelMatrix)
{
	if (!_loaded)
	{
		bool success = false, binaryExists = false;

		if (_useBinary && (binaryExists = std::filesystem::exists(_filename + BINARY_EXTENSION)))
		{
			success = this->loadModelFromBinaryFile();
		}

		if (!success)
		{
			success = this->loadModelFromPLY(modelMatrix);
		}

		std::cout << "Number of Points: " << _points.size() << std::endl;

		if (success && !binaryExists)
		{
			this->writeToBinary(_filename + BINARY_EXTENSION);
		}

		_loaded = true;
		
		return true;
	}

	return false;
}

bool PointCloud::writePointCloud(const std::string& filename, const bool ascii)
{
	std::thread writePointCloudThread(&PointCloud::threadedWritePointCloud, this, filename, ascii);
	writePointCloudThread.detach();

	return true;
}

/// [Protected methods]

void PointCloud::computeCloudData()
{
	ModelComponent* modelComp = _modelComp[0];

	// Fill point cloud indices with iota
	modelComp->_pointCloud.resize(_points.size());
	std::iota(modelComp->_pointCloud.begin(), modelComp->_pointCloud.end(), 0);
}

void PointCloud::getLabels(std::shared_ptr<tinyply::PlyData>& plyLabels, std::vector<PointModel>& points)
{
	const bool isDouble = plyLabels->t == tinyply::Type::FLOAT64, isUchar = plyLabels->t == tinyply::Type::UINT8, isFloat = plyLabels->t == tinyply::Type::FLOAT32;
	const size_t numPoints = plyLabels->count;
	const size_t numPointsBytes = numPoints * (isDouble ? sizeof(double) : (isFloat ? sizeof(float) : sizeof(uint8_t)));

	float* labelsRawFloat = nullptr;
	double* labelsRawDouble = nullptr;
	uint8_t* labelsRawUChar = nullptr;

	if (isDouble)
	{
		labelsRawDouble = new double[numPoints];
		std::memcpy(labelsRawDouble, plyLabels->buffer.get(), numPointsBytes);
	}
	else if (isUchar)
	{
		labelsRawUChar = new uint8_t[numPoints];
		std::memcpy(labelsRawUChar, plyLabels->buffer.get(), numPointsBytes);
	}
	else
	{
		labelsRawFloat = new float[numPoints];
		std::memcpy(labelsRawFloat, plyLabels->buffer.get(), numPointsBytes);
	}

	if (isDouble)
	{
		for (unsigned index = 0; index < numPoints; ++index)
		{
			_points[index]._label = labelsRawDouble[index];
			_maxLabel = std::max(_points[index]._label, _maxLabel);
		}
	}
	else if (isUchar)
	{
		for (unsigned index = 0; index < numPoints; ++index) 
		{
			_points[index]._label = labelsRawUChar[index];
			_maxLabel = std::max(_points[index]._label, _maxLabel);
		}
	}
	else
	{
		for (unsigned index = 0; index < numPoints; ++index)
		{
			_points[index]._label = labelsRawFloat[index];
			_maxLabel = std::max(_points[index]._label, _maxLabel);
		}
	}

	delete[] labelsRawDouble;
	delete[] labelsRawFloat;
	delete[] labelsRawUChar;
}

void PointCloud::getPoints(std::shared_ptr<tinyply::PlyData>& plyPoints, std::vector<PointModel>& points)
{
	const bool isDouble = plyPoints->t == tinyply::Type::FLOAT64;
	const size_t numPoints = plyPoints->count;
	const size_t numPointsBytes = numPoints * (!isDouble ? sizeof(float) : sizeof(double)) * 3;

	float* pointsRawFloat = nullptr;
	double* pointsRawDouble = nullptr;
	unsigned baseIndex;

	if (!isDouble)
	{
		pointsRawFloat = new float[numPoints * 3];
		std::memcpy(pointsRawFloat, plyPoints->buffer.get(), numPointsBytes);
	}
	else
	{
		pointsRawDouble = new double[numPoints * 3];
		std::memcpy(pointsRawDouble, plyPoints->buffer.get(), numPointsBytes);
	}

	_points.clear();
	_points.resize(numPoints);

	if (!isDouble)
	{
		for (unsigned index = 0; index < numPoints; ++index)
		{
			baseIndex = index * 3;
			_points[index]._point = vec3(pointsRawFloat[baseIndex], pointsRawFloat[baseIndex + 1], pointsRawFloat[baseIndex + 2]);
			_aabb.update(_points[index]._point);
		}
	}
	else
	{
		for (unsigned index = 0; index < numPoints; ++index)
		{
			baseIndex = index * 3;
			_points[index]._point = vec3(pointsRawDouble[baseIndex], pointsRawDouble[baseIndex + 1], pointsRawDouble[baseIndex + 2]);
			_aabb.update(_points[index]._point);
		}
	}

	delete[] pointsRawFloat;
	delete[] pointsRawDouble;
}

bool PointCloud::loadModelFromBinaryFile()
{
	return this->readBinary(_filename + BINARY_EXTENSION, _modelComp);
}

bool PointCloud::loadModelFromPLY(const mat4& modelMatrix)
{
	std::unique_ptr<std::istream> fileStream;
	std::vector<uint8_t> byteBuffer;
	std::shared_ptr<tinyply::PlyData> plyPoints, plyLabels;
	unsigned baseIndex;
	float* pointsRawFloat = nullptr;
	double* pointsRawDouble = nullptr;
	uint8_t* colorsRaw;

	try
	{
		const std::string filename = _filename + PLY_EXTENSION;
		fileStream.reset(new std::ifstream(filename, std::ios::binary));

		if (!fileStream || fileStream->fail()) return false;

		fileStream->seekg(0, std::ios::end);
		const float size_mb = fileStream->tellg() * float(1e-6);
		fileStream->seekg(0, std::ios::beg);

		tinyply::PlyFile file;
		file.parse_header(*fileStream);

		try { plyPoints = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
		catch (const std::exception& e) { return false; }
		
		try { plyLabels = file.request_properties_from_element("vertex", { "scalar_Classification" }); }
		catch (const std::exception & e) 
		{ 
			try {
				plyLabels = file.request_properties_from_element("vertex", { "semanticGroup" });
			}
			catch (const std::exception& e)
			{
				return false;
			}
		}

		file.read(*fileStream);

		{
			const bool isDouble = plyPoints->t == tinyply::Type::FLOAT64;
			const size_t numPoints = plyPoints->count;
			const size_t numPointsBytes = numPoints * (!isDouble ? sizeof(float) : sizeof(double)) * 3;
			const size_t numLabelBytes = numPoints * sizeof(float);

			// Allocate space
			this->getPoints(plyPoints, _points);
			this->getLabels(plyLabels, _points);
		}
	}
	catch (const std::exception & e)
	{
		std::cerr << "Caught tinyply exception: " << e.what() << std::endl;

		return false;
	}

	return true;
}

bool PointCloud::readBinary(const std::string& filename, const std::vector<Model3D::ModelComponent*>& modelComp)
{
	std::ifstream fin(filename, std::ios::in | std::ios::binary);
	if (!fin.is_open())
	{
		return false;
	}

	size_t numPoints;

	fin.read((char*)&numPoints, sizeof(size_t));
	_points.resize(numPoints);
	fin.read((char*)&_points[0], numPoints * sizeof(PointModel));
	fin.read((char*)&_aabb, sizeof(AABB));
	fin.read((char*)&_maxLabel, sizeof(unsigned));

	fin.close();

	return true;
}

void PointCloud::setVAOData()
{
	VAO* vao = new VAO(false);
	ModelComponent* modelComp = _modelComp[0];
	unsigned startIndex = 0, size = modelComp->_pointCloud.size(), currentSize;

	vao->setVBOData(RendEnum::VBO_POSITION, _points, GL_STATIC_DRAW);
	vao->setIBOData(RendEnum::IBO_POINT_CLOUD, modelComp->_pointCloud);
	modelComp->_topologyIndicesLength[RendEnum::IBO_POINT_CLOUD] = unsigned(modelComp->_pointCloud.size());
}

void PointCloud::threadedWritePointCloud(const std::string& filename, const bool ascii)
{
	std::filebuf fileBuffer;					
	fileBuffer.open(WRITE_POINT_CLOUD_FOLDER + filename, ascii ? std::ios::out : std::ios::out | std::ios::binary);

	std::ostream outstream(&fileBuffer);
	if (outstream.fail()) throw std::runtime_error("Failed to open " + filename + ".");

	tinyply::PlyFile pointCloud;

	std::vector<vec3> position;
	std::vector<uint8_t> labels;

	for (int pointIdx = 0; pointIdx < _points.size(); ++pointIdx)
	{
		position.push_back(_points[pointIdx]._point);
		labels.push_back(_points[pointIdx]._label);
	}

	const std::string componentName = "pointCloud";
	pointCloud.add_properties_to_element(componentName, { "x", "y", "z" }, tinyply::Type::FLOAT32, position.size(), reinterpret_cast<uint8_t*>(position.data()), tinyply::Type::INVALID, 0);
	pointCloud.add_properties_to_element(componentName, { "class" }, tinyply::Type::UINT8, labels.size(), reinterpret_cast<uint8_t*>(labels.data()), tinyply::Type::INVALID, 0);
	pointCloud.write(outstream, !ascii);
}

bool PointCloud::writeToBinary(const std::string& filename)
{
	std::ofstream fout(filename, std::ios::out | std::ios::binary);
	if (!fout.is_open())
	{
		return false;
	}

	const size_t numPoints = _points.size();
	fout.write((char*)&numPoints, sizeof(size_t));
	fout.write((char*)&_points[0], numPoints * sizeof(PointModel));
	fout.write((char*)&_aabb, sizeof(AABB));
	fout.write((char*)&_maxLabel, sizeof(unsigned));

	fout.close();

	return true;
}