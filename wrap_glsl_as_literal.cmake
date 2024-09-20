function(make_includable input_file output_file)
    file(READ ${input_file} content)
    set(content "R\"(\n${content})\"")
    file(WRITE ${output_file} "${content}")
endfunction()

# Set the source directory
set(SHADER_SRC_DIR "${SRC_DIR}/shaders")

# Remove ghost .str files (those without corresponding .glsl files)
file(GLOB STR_FILES "${SHADER_SRC_DIR}/*.str")
foreach(STR_FILE ${STR_FILES})
    get_filename_component(FILE_BASENAME ${STR_FILE} NAME_WE)
    set(GLSL_FILE "${SHADER_SRC_DIR}/${FILE_BASENAME}.glsl")

    if(NOT EXISTS ${GLSL_FILE})
        message(STATUS "Removing ghost file ${FILE_BASENAME}.str")
        file(REMOVE ${STR_FILE})
    endif()
endforeach()

# Check .glsl files and update .str files if necessary
file(GLOB GLSL_FILES "${SHADER_SRC_DIR}/*.glsl")
foreach(GLSL_FILE ${GLSL_FILES})
    get_filename_component(FILE_BASENAME ${GLSL_FILE} NAME_WE)
    set(STR_FILE "${SHADER_SRC_DIR}/${FILE_BASENAME}.str")

    if(NOT EXISTS ${STR_FILE})
        message(STATUS "Generating  ${FILE_BASENAME}.str")
        make_includable(${GLSL_FILE} ${STR_FILE})
    else()
        # Compare timestamps and regenerate .str if .glsl is newer
        file(TIMESTAMP ${GLSL_FILE} GLSL_TIMESTAMP UTC)
        file(TIMESTAMP ${STR_FILE} STR_TIMESTAMP UTC)

        if(GLSL_TIMESTAMP STRGREATER STR_TIMESTAMP)
            message(STATUS "Updating ${FILE_BASENAME}.str")
            make_includable(${GLSL_FILE} ${STR_FILE})
        endif()
    endif()
endforeach()