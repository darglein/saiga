SUBDIRS = doc # lib-config-src # man
ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libhello.la

libhello_la_SOURCES = 	src/camera/camera.cpp \
			src/camera/controlable_camera.cpp \
			\
			src/opengl/framebuffer.cpp \
			src/opengl/mesh.cpp \
			src/opengl/objloader.cpp \
			src/opengl/texture.cpp \
			src/opengl/shader.cpp \
			src/opengl/vertexBuffer.cpp \
			src/opengl/indexedVertexBuffer.cpp \
			src/opengl/indexBuffer.cpp \
			src/opengl/mesh_object.cpp \
			src/opengl/simple_mesh_objects.cpp \
			\
			src/rendering/renderer.cpp \
			src/rendering/deferred_renderer.cpp \
			src/rendering/material.cpp \
			src/rendering/object3d.cpp \
			\
			src/rendering/lighting/deferred_lighting.cpp \
			src/rendering/lighting/directional_light.cpp \
			src/rendering/lighting/point_light.cpp \
			src/rendering/lighting/spot_light.cpp \
			src/rendering/lighting/light.cpp \
			src/rendering/lighting/lighting_controller.cpp \
			\
			src/rendering/overlay/overlay.cpp \
			src/rendering/overlay/console.cpp \
			\
			src/rendering/particles/particle.cpp \
			src/rendering/particles/particle_shader.cpp \
			src/rendering/particles/particlesystem.cpp \
			\
			src/sdl/timer.cpp \
			src/sdl/sdl_eventhandler.cpp \
			\
			src/util/glm.cpp \
			src/util/fpscounter.cpp \
			src/util/png_wrapper.cpp \
			src/util/inputcontroller.cpp \
			src/util/keyboardbinds.cpp \
			src/util/observer.cpp \
			\
			src/window/sdl_window.cpp \
			src/window/glfw_window.cpp \
			src/window/window.cpp \
			src/window/glfw_eventhandler.cpp \
			\
			src/geometry/aabb.cpp \
			src/geometry/sphere.cpp \
			src/geometry/plane.cpp \
			src/geometry/ray.cpp \
			src/geometry/grid.cpp \
			src/geometry/triangle_mesh.cpp \
			src/geometry/triangle_mesh_generator.cpp \
			src/geometry/cone.cpp \
			\
			src/text/text.cpp \
			src/text/dynamic_text.cpp \
			src/text/text_generator.cpp 
			
			
library_includedir = $(includedir)/$(PACKAGE)-$(VERSION)


nobase_include_HEADERS = $(PACKAGE)/camera/camera.h \
			$(PACKAGE)/camera/controlable_camera.h \
			\
			$(PACKAGE)/opengl/framebuffer.h \
			$(PACKAGE)/opengl/mesh.h \
			$(PACKAGE)/opengl/objloader.h \
			$(PACKAGE)/opengl/texture.h \
			$(PACKAGE)/opengl/shader.h \
			$(PACKAGE)/opengl/vertexBuffer.h \
			$(PACKAGE)/opengl/indexBuffer.h \
			$(PACKAGE)/opengl/indexedVertexBuffer.h \
			$(PACKAGE)/opengl/vertex.h \
			$(PACKAGE)/opengl/mesh_object.h \
			$(PACKAGE)/opengl/simple_mesh_objects.h \
			\
			$(PACKAGE)/rendering/renderer.h \
			$(PACKAGE)/rendering/deferred_renderer.h \
			$(PACKAGE)/rendering/material.h \
			$(PACKAGE)/rendering/object3d.h \
			\
			$(PACKAGE)/rendering/lighting/deferred_lighting.h \
			$(PACKAGE)/rendering/lighting/directional_light.h \
			$(PACKAGE)/rendering/lighting/point_light.h \
			$(PACKAGE)/rendering/lighting/spot_light.h \
			$(PACKAGE)/rendering/lighting/light.h \
			$(PACKAGE)/rendering/lighting/lighting_controller.h \
			\
			$(PACKAGE)/rendering/particles/particle.h \
			$(PACKAGE)/rendering/particles/particle_shader.h \
			$(PACKAGE)/rendering/particles/particlesystem.h \
			\
			$(PACKAGE)/rendering/overlay/overlay.h \
			$(PACKAGE)/rendering/overlay/console.h \
			\
			$(PACKAGE)/sdl/timer.h \
			$(PACKAGE)/sdl/sdl_eventhandler.h \
			$(PACKAGE)/sdl/listener.h \
			\
			$(PACKAGE)/util/glm.h \
			$(PACKAGE)/util/fpscounter.h \
			$(PACKAGE)/util/png_wrapper.h \
			$(PACKAGE)/util/loader.h \
			$(PACKAGE)/util/error.h \
			$(PACKAGE)/util/inputcontroller.h \
			$(PACKAGE)/util/keyboardbinds.h \
			$(PACKAGE)/util/observer.h \
			\
			$(PACKAGE)/window/sdl_window.h \
			$(PACKAGE)/window/glfw_window.h \
			$(PACKAGE)/window/window.h \
			$(PACKAGE)/window/glfw_eventhandler.h \
			\
			$(PACKAGE)/geometry/aabb.h \
			$(PACKAGE)/geometry/sphere.h \
			$(PACKAGE)/geometry/plane.h \
			$(PACKAGE)/geometry/ray.h \
			$(PACKAGE)/geometry/grid.h \
			$(PACKAGE)/geometry/triangle_mesh.h \
			$(PACKAGE)/geometry/triangle_mesh_generator.h \
			$(PACKAGE)/geometry/cone.h \
			\
			$(PACKAGE)/text/text.h \
			$(PACKAGE)/text/dynamic_text.h \
			$(PACKAGE)/text/text_generator.h


libhello_la_CPPFLAGS = 	-D PACKAGE=\"$(PACKAGE)\" -D VERSION=\"$(VERSION)\" \
			-I$(top_srcdir)/$(PACKAGE) \
			-I$(top_srcdir) $(GLFW_CFLAGS)
			

libhello_la_LIBADD =  $(GLFW_LIBS)
			
library_nonversioned_includedir = $(includedir)/$(PACKAGE)



.PHONY: doc
doc:
	cd doc; make doc

