# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1

# Utility rule file for debug.

# Include the progress variables for this target.
include CMakeFiles/debug.dir/progress.make

CMakeFiles/debug:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Switching CMAKE_BUILD_TYPE to Debug"
	/usr/bin/cmake -DCMAKE_BUILD_TYPE=Debug /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1
	/usr/bin/cmake -E echo "***"
	/usr/bin/cmake -E echo "*** Switched to Debug mode. Now recompile with:  \$$ make"
	/usr/bin/cmake -E echo "***"

debug: CMakeFiles/debug
debug: CMakeFiles/debug.dir/build.make

.PHONY : debug

# Rule to build all files generated by this target.
CMakeFiles/debug.dir/build: debug

.PHONY : CMakeFiles/debug.dir/build

CMakeFiles/debug.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/debug.dir/cmake_clean.cmake
.PHONY : CMakeFiles/debug.dir/clean

CMakeFiles/debug.dir/depend:
	cd /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1 /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1 /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1 /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1 /home/vikram/dealii_source/dealii-9.3.3/examples/mtech_1/CMakeFiles/debug.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/debug.dir/depend

