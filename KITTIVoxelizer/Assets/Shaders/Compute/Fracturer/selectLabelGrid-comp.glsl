#version 450

#extension GL_ARB_compute_variable_group_size: enable
#extension GL_NV_gpu_shader5 : enable

layout (local_size_variable) in;

#include <Assets/Shaders/Compute/Templates/modelStructs.glsl>

layout (std430, binding = 0) buffer GridBuffer		{ uint16_t			grid[]; };
layout (std430, binding = 1) buffer LabelBuffer		{ uint				label[]; };

uniform uint numCells;
uniform uint numLabels;

void main()
{
	const uint index = gl_GlobalInvocationID.x;
	if (index >= numCells * numLabels) return;

	uint maxOccurrence = 0;
	uint baseIndex = index * numLabels, maxOccurrLabel = baseIndex;

	for (uint idx = baseIndex; idx < baseIndex + numLabels; ++idx)
	{
		if (label[idx] > maxOccurrence)
		{
			maxOccurrence = label[idx];
			maxOccurrLabel = idx - baseIndex;
		}
	}

	if (maxOccurrence > 0)
	{
		grid[index] = uint16_t(maxOccurrLabel);
	}
}