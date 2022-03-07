#include "stdafx.h"
#include "RegularGrid.h"

#include "Graphics/Core/OpenGLUtilities.h"
#include "Graphics/Core/ShaderList.h"
#include "tinyply/tinyply.h"

/// Public methods

RegularGrid::RegularGrid(const AABB& aabb, uvec3 subdivisions) :
	_aabb(aabb), _numDivs(subdivisions)
{
	_cellSize = vec3((_aabb.max().x - _aabb.min().x) / float(subdivisions.x), (_aabb.max().y - _aabb.min().y) / float(subdivisions.y), (_aabb.max().z - _aabb.min().z) / float(subdivisions.z));

	this->buildGrid();
}

RegularGrid::RegularGrid(uvec3 subdivisions) : _numDivs(subdivisions)
{
	
}

RegularGrid::~RegularGrid()
{
}

void RegularGrid::exportBinary(const std::string& filename)
{
	std::string outputfilename = filename + BINARY_EXTENSION;

	std::ofstream out(outputfilename.c_str());
	std::vector<uint8_t> packed = this->pack(_grid);
	out.write((const char*)&packed[0], packed.size() * sizeof(uint8_t));
	out.close();
}

void RegularGrid::fill(PointCloud* pointCloud)
{
	ComputeShader* resetShader = ShaderList::getInstance()->getComputeShader(RendEnum::RESET_LABEL_INDEX);
	ComputeShader* boundaryShader = ShaderList::getInstance()->getComputeShader(RendEnum::BUILD_REGULAR_GRID);
	ComputeShader* selectLabelShader = ShaderList::getInstance()->getComputeShader(RendEnum::SELECT_LABEL_GRID);

	// Input data
	uvec3 numDivs		= this->getNumSubdivisions();
	unsigned numCells	= numDivs.x * numDivs.y * numDivs.z;
	unsigned numThreads = pointCloud->getNumberOfPoints();
	unsigned numLabels	= pointCloud->getMaxLabel() + 1;
	unsigned numGroups1	= ComputeShader::getNumGroups(numThreads);
	unsigned numGroups2 = ComputeShader::getNumGroups(numCells);
	unsigned numGroups3 = ComputeShader::getNumGroups(numCells * numLabels);

	// Input data
	const GLuint vertexSSBO = ComputeShader::setReadBuffer(*pointCloud->getPoints(), GL_STATIC_DRAW);
	const GLuint gridSSBO	= ComputeShader::setReadBuffer(&_grid[0], numCells, GL_DYNAMIC_DRAW);
	const GLuint labelSSBO = ComputeShader::setWriteBuffer(unsigned(), numCells * numLabels, GL_DYNAMIC_DRAW);

	// 1. Init to zero the label buffer
	resetShader->bindBuffers(std::vector<GLuint>{ labelSSBO });
	resetShader->use();
	resetShader->setUniform("numLabels", GLuint(numCells * numLabels));
	resetShader->execute(numGroups3, 1, 1, ComputeShader::getMaxGroupSize(), 1, 1);

	boundaryShader->bindBuffers(std::vector<GLuint>{ vertexSSBO, gridSSBO, labelSSBO });
	boundaryShader->use();
	boundaryShader->setUniform("aabbMin", _aabb.min());
	boundaryShader->setUniform("cellSize", _cellSize);
	boundaryShader->setUniform("gridDims", numDivs);
	boundaryShader->setUniform("numLabels", numLabels);
	boundaryShader->setUniform("numPoints", numThreads);
	boundaryShader->execute(numGroups1, 1, 1, ComputeShader::getMaxGroupSize(), 1, 1);

	selectLabelShader->bindBuffers(std::vector<GLuint>{ gridSSBO, labelSSBO });
	selectLabelShader->use();
	selectLabelShader->setUniform("numLabels", numLabels);
	selectLabelShader->setUniform("numCells", numCells);
	selectLabelShader->execute(numGroups2, 1, 1, ComputeShader::getMaxGroupSize(), 1, 1);

	uint16_t* gridData = ComputeShader::readData(gridSSBO, uint16_t());
	_grid = std::vector<uint16_t>(gridData, gridData + numCells);

	GLuint buffers[] = { vertexSSBO, gridSSBO, labelSSBO };
	glDeleteBuffers(sizeof(buffers) / sizeof(GLuint), buffers);
}

void RegularGrid::getAABBs(std::vector<AABB>& aabb)
{
	vec3 max, min;
	
	for (int x = 0; x < _numDivs.x; ++x)
	{
		for (int y = 0; y < _numDivs.y; ++y)
		{
			for (int z = 0; z < _numDivs.z; ++z)
			{
				if (_grid[this->getPositionIndex(x, y, z)] != VOXEL_EMPTY)
				{
					min = _aabb.min() + _cellSize * vec3(x, y, z);
					max = min + _cellSize;

					aabb.push_back(AABB(min, max));
				}
			}
		}
	}
}

void RegularGrid::insertPoint(const vec3& position, unsigned index)
{
	uvec3 gridIndex = getPositionIndex(position);

	_grid[this->getPositionIndex(gridIndex.x, gridIndex.y, gridIndex.z)] = index;
}

void RegularGrid::queryCluster(const std::vector<Model3D::VertexGPUData>& vertices, const std::vector<Model3D::FaceGPUData>& faces, std::vector<float>& clusterIdx)
{
	ComputeShader* shader = ShaderList::getInstance()->getComputeShader(RendEnum::ASSIGN_FACE_CLUSTER);

	// Input data
	uvec3 numDivs = this->getNumSubdivisions();
	unsigned numThreads = vertices.size();
	unsigned numGroups = ComputeShader::getNumGroups(numThreads);

	// Input data
	const GLuint vertexSSBO = ComputeShader::setReadBuffer(vertices, GL_STATIC_DRAW);
	const GLuint faceSSBO = ComputeShader::setReadBuffer(faces, GL_DYNAMIC_DRAW);
	const GLuint gridSSBO = ComputeShader::setReadBuffer(_grid, GL_STATIC_DRAW);
	const GLuint clusterSSBO = ComputeShader::setWriteBuffer(float(), vertices.size(), GL_DYNAMIC_DRAW);

	shader->bindBuffers(std::vector<GLuint>{ vertexSSBO, faceSSBO, gridSSBO, clusterSSBO });
	shader->use();
	shader->setUniform("aabbMin", _aabb.min());
	shader->setUniform("cellSize", _cellSize);
	shader->setUniform("gridDims", numDivs);
	shader->setUniform("numVertices", GLuint(vertices.size()));
	shader->execute(numGroups, 1, 1, ComputeShader::getMaxGroupSize(), 1, 1);

	float* clusterData = ComputeShader::readData(clusterSSBO, float());
	clusterIdx = std::vector<float>(clusterData, clusterData + vertices.size());

	GLuint buffers[] = { vertexSSBO, faceSSBO, gridSSBO, clusterSSBO };
	glDeleteBuffers(sizeof(buffers) / sizeof(GLuint), buffers);
}

// [Protected methods]

uint16_t* RegularGrid::data()
{
	return _grid.data();
}

uint16_t RegularGrid::at(int x, int y, int z) const
{
	return _grid[this->getPositionIndex(x, y, z)];
}

glm::uvec3 RegularGrid::getNumSubdivisions() const
{
	return _numDivs;
}

void RegularGrid::homogenize()
{
	for (unsigned int x = 0; x < _numDivs.x; ++x)
		for (unsigned int y = 0; y < _numDivs.y; ++y)
			for (unsigned int z = 0; z < _numDivs.z; ++z)
				if (this->at(x, y, z) != VOXEL_EMPTY)
					this->set(x, y, z, VOXEL_FREE);
}

bool RegularGrid::isOccupied(int x, int y, int z) const
{
	return this->at(x, y, z) != VOXEL_EMPTY;
}

bool RegularGrid::isEmpty(int x, int y, int z) const
{
	return this->at(x, y, z) == VOXEL_EMPTY;
}

size_t RegularGrid::length() const
{
	return _numDivs.x * _numDivs.y * _numDivs.z;
}

void RegularGrid::set(int x, int y, int z, uint8_t i)
{
	_grid[this->getPositionIndex(x, y, z)] = i;
}

/// Protected methods	

void RegularGrid::buildGrid()
{	
	_grid = std::vector<uint16_t>(_numDivs.x * _numDivs.y * _numDivs.z);
	std::fill(_grid.begin(), _grid.end(), VOXEL_EMPTY);
}

uvec3 RegularGrid::getPositionIndex(const vec3& position)
{
	unsigned x = (position.x - _aabb.min().x) / _cellSize.x, y = (position.y - _aabb.min().y) / _cellSize.y, z = (position.z - _aabb.min().z) / _cellSize.z;
	unsigned zeroUnsigned = 0;

	return uvec3(glm::clamp(x, zeroUnsigned, _numDivs.x - 1), glm::clamp(y, zeroUnsigned, _numDivs.y - 1), glm::clamp(z, zeroUnsigned, _numDivs.z - 1));
}

unsigned RegularGrid::getPositionIndex(int x, int y, int z) const
{
	return x * _numDivs.y * _numDivs.z + y * _numDivs.z + z;
}

unsigned RegularGrid::getPositionIndex(int x, int y, int z, const uvec3& numDivs)
{
	return x * numDivs.y * numDivs.z + y * numDivs.z + z;
}
