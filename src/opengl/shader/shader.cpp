#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/texture/raw_texture.h"
#include "saiga/util/error.h"
#include <fstream>
#include <algorithm>
#include "saiga/util/assert.h"

GLuint Shader::boundShader = 0;

Shader::Shader(){
}

Shader::~Shader(){
	//cout << "~Shader " << name << endl;
	destroyProgram();
}

// ===================================== program stuff =====================================


GLuint Shader::createProgram(){

	program = glCreateProgram();

	//attach all shaders
	for (auto& sp : shaders){
//        cout << "Attaching shader " << sp->id << endl;
		glAttachShader(program, sp->id);
	}
	assert_no_glerror();
	glLinkProgram(program);
    assert_no_glerror();
	if (!printProgramLog()){
        //do not assert here, because printprogramlog will sometimes only print warnings.
//        SAIGA_SAIGA_ASSERT(0);
//		return 0;
	}

	assert_no_glerror();
	for (auto& sp : shaders){
		sp->deleteGLShader();
	}

	assert_no_glerror();

	checkUniforms();

	assert_no_glerror();
	return program;
}

void Shader::destroyProgram()
{
	if (program != 0){
		glDeleteProgram(program);
		program = 0;
       // assert_no_glerror();
	}
}

bool Shader::printProgramLog(){
	//Make sure name is shader
	if (glIsProgram(program) == GL_TRUE)
	{
		//Program log length
		int infoLogLength = 0;
		int maxLength = infoLogLength;

		//Get info std::string length
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

		//Allocate std::string
		char* infoLog = new char[maxLength];

		//Get info log
		glGetProgramInfoLog(program, maxLength, &infoLogLength, infoLog);
		if (infoLogLength > 0)
		{
			//Print Log
            std::cout << "Program info log:" << std::endl;
			std::cout << infoLog << std::endl;
		}
        assert_no_glerror();

		//Deallocate std::string
		delete[] infoLog;
		return infoLogLength == 0;
	}
	else
	{
		cout << "Name " << program << " is not a program" << endl;
		return false;
	}
}

void Shader::bind(){
    SAIGA_ASSERT(program);
	//allow double bind
    SAIGA_ASSERT(boundShader == program || boundShader == 0);
	boundShader = program;
	glUseProgram(program);
    assert_no_glerror();
}

void Shader::unbind(){
	//allow double unbind
    SAIGA_ASSERT(boundShader == program || boundShader == 0);
	boundShader = 0;
#if defined(SAIGA_DEBUG)
	glUseProgram(0);
#endif
    assert_no_glerror();
}

bool Shader::isBound(){
	return boundShader == program;
}

// ===================================== Compute Shaders =====================================


glm::uvec3 Shader::getComputeWorkGroupSize()
{
	GLint work_size[3];
	glGetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, work_size);
    assert_no_glerror();
	return glm::uvec3(work_size[0], work_size[1], work_size[2]);
}

glm::uvec3 Shader::getNumGroupsCeil(const glm::uvec3 &problem_size)
{
	return getNumGroupsCeil(problem_size, getComputeWorkGroupSize());
}

glm::uvec3 Shader::getNumGroupsCeil(const glm::uvec3 &problem_size, const glm::uvec3 &work_group_size)
{
	//    glm::uvec3 ret = problem_size/work_group_size;
	//    glm::uvec3 rest = problem_size%work_group_size;

	//    ret.x += rest.x ? 1 : 0;
	//    ret.y += rest.y ? 1 : 0;
	//    ret.z += rest.z ? 1 : 0;

	//    return ret;

	return (problem_size + work_group_size - glm::uvec3(1)) / (work_group_size);
}

void Shader::dispatchCompute(const glm::uvec3 &num_groups)
{
    SAIGA_ASSERT(isBound());
	dispatchCompute(num_groups.x, num_groups.y, num_groups.z);
    assert_no_glerror();
}

void Shader::dispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    SAIGA_ASSERT(isBound());
	glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    assert_no_glerror();
}



void Shader::memoryBarrier(MemoryBarrierMask barriers)
{
	glMemoryBarrier(barriers);
    assert_no_glerror();
}

void Shader::memoryBarrierTextureFetch()
{
	memoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    assert_no_glerror();
}

void Shader::memoryBarrierImageAccess()
{
	memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    assert_no_glerror();
}

// ===================================== uniforms =====================================


GLint Shader::getUniformLocation(const char* name){
	GLint i = glGetUniformLocation(program, name);
	return i;
}

void Shader::getUniformInfo(GLuint location){
	const GLsizei bufSize = 128;

	GLsizei length;
	GLint size;
	GLenum type;
	GLchar name[bufSize];

	glGetActiveUniform(program, location, bufSize, &length, &size, &type, name);

	cout << "uniform info " << location << endl;
	cout << "name " << name << endl;
	//    cout<<"length "<<length<<endl;
	cout << "size " << size << endl;
	cout << "type " << type << endl;
}

// ===================================== uniform blocks =====================================

GLuint Shader::getUniformBlockLocation(const char *name)
{
	GLuint blockIndex = glGetUniformBlockIndex(program, name);

	if (blockIndex == GL_INVALID_INDEX){
		std::cerr << "glGetUniformBlockIndex: uniform block invalid!" << endl;
	}
	return blockIndex;
}

void Shader::setUniformBlockBinding(GLuint blockLocation, GLuint bindingPoint)
{
	glUniformBlockBinding(program, blockLocation, bindingPoint);
}

GLint Shader::getUniformBlockSize(GLuint blockLocation)
{
	GLint ret;
	glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_DATA_SIZE, &ret);
	return ret;
}

std::vector<GLint> Shader::getUniformBlockIndices(GLuint blockLocation)
{
	GLint ret;
	glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &ret);

	std::vector<GLint> indices(ret);
	glGetActiveUniformBlockiv(program, blockLocation, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &indices[0]);

	return indices;
}

std::vector<GLint> Shader::getUniformBlockSize(std::vector<GLint> indices)
{
	std::vector<GLint> ret(indices.size());
	glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_SIZE, ret.data());
	return ret;
}

std::vector<GLint> Shader::getUniformBlockType(std::vector<GLint> indices)
{
	std::vector<GLint> ret(indices.size());
	glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_TYPE, ret.data());
	return ret;
}

std::vector<GLint> Shader::getUniformBlockOffset(std::vector<GLint> indices)
{
	std::vector<GLint> ret(indices.size());
	glGetActiveUniformsiv(program, indices.size(), (GLuint*)indices.data(), GL_UNIFORM_OFFSET, ret.data());
	return ret;
}

// ===================================== uniform uploads =====================================


void Shader::upload(int location, const mat4 &m){
    SAIGA_ASSERT(isBound());
	glUniformMatrix4fv(location, 1, GL_FALSE, (GLfloat*)&m[0]);
    assert_no_glerror();
}

void Shader::upload(int location, const vec4 &v){
    SAIGA_ASSERT(isBound());
	glUniform4fv(location, 1, (GLfloat*)&v[0]);
    assert_no_glerror();
}

void Shader::upload(int location, const vec3 &v){
    SAIGA_ASSERT(isBound());
	glUniform3fv(location, 1, (GLfloat*)&v[0]);
    assert_no_glerror();
}

void Shader::upload(int location, const vec2 &v){
    SAIGA_ASSERT(isBound());
	glUniform2fv(location, 1, (GLfloat*)&v[0]);
    assert_no_glerror();
}

void Shader::upload(int location, const int &i){
    SAIGA_ASSERT(isBound());
	glUniform1i(location, (GLint)i);
    assert_no_glerror();
}

void Shader::upload(int location, const float &f){
    SAIGA_ASSERT(isBound());
	glUniform1f(location, (GLfloat)f);
    assert_no_glerror();
}


void Shader::upload(int location, int count, mat4* m){
    SAIGA_ASSERT(isBound());
	glUniformMatrix4fv(location, count, GL_FALSE, (GLfloat*)m);
    assert_no_glerror();
}

void Shader::upload(int location, int count, vec4* v){
    SAIGA_ASSERT(isBound());
	glUniform4fv(location, count, (GLfloat*)v);
    assert_no_glerror();
}

void Shader::upload(int location, int count, vec3* v){
    SAIGA_ASSERT(isBound());
	glUniform3fv(location, count, (GLfloat*)v);
    assert_no_glerror();
}

void Shader::upload(int location, int count, vec2* v){
    SAIGA_ASSERT(isBound());
	glUniform2fv(location, count, (GLfloat*)v);
    assert_no_glerror();
}

void Shader::upload(int location, raw_Texture *texture, int textureUnit)
{
    SAIGA_ASSERT(isBound());
	texture->bind(textureUnit);
	Shader::upload(location, textureUnit);
    assert_no_glerror();
}
