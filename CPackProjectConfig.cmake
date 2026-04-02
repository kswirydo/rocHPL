
    # Generator-specific configuration for consistent archive paths
    if(CPACK_GENERATOR MATCHES "TGZ|ZIP")
        # Use relative paths for archive generators (consistent across platforms)
        set(CPACK_SET_DESTDIR OFF)
        set(CPACK_PACKAGING_INSTALL_PREFIX "/")
    endif()
    