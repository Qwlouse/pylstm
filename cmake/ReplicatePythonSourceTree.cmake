# Note: when executed in the build dir, then CMAKE_CURRENT_SOURCE_DIR is the
# build dir.
IF (EXISTS "setup.py")
	FILE(COPY setup.py DESTINATION "${CMAKE_ARGV3}")
ENDIF (EXISTS "setup.py")

FILE(COPY src test bin DESTINATION "${CMAKE_ARGV3}"
     FILES_MATCHING PATTERN "*.py")
