import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger('pythia.util')

def image2sections(image, section_size=100):
    """Create sections of the image.

    Parameters
    ----------
    image : Numpy array
        Image to be split into sections
    section_size: int
        Number of pixels to use in each section

    Returns
    -------
    sections
        List with sections of the image
    """
    sections = []
    i = 0
    j = 0
    
    while (j + 1) * section_size < image.shape[0]:
        while (i + 1) * section_size < image.shape[1]:
            sections.append(
                image[
                    j * 100:(j + 1) * 100,
                    i * 100:(i + 1) * 100
                ]
            )
            i = i + 1
        i = 0
        j = j + 1
        
    return sections

def image2sections_and_classes(image, classes, section_size=100):
    """Create sections of the image.

    Parameters
    ----------
    image : Numpy array
        Image to be split into sections
    section_size: int
        Number of pixels to use in each section

    Returns
    -------
    sections
        List with sections of the image
    """
    sections = []
    i = 0
    j = 0
    
    # How the image is segmented.
    # +---+---+---+
    # | 1 | 2 | 3 |
    # +---+---+---+
    # | 4 | 5 | 6 |
    # +---+---+---+
    # | 7 | 8 | 9 |
    # +---+---+---+
    while (i + 1) * section_size < image.shape[0]:
        i_floor = i * 100
        i_ceil = (i + 1) * 100
        while (j + 1) * section_size < image.shape[1]:
            j_floor = j * 100
            j_ceil = (j + 1) * 100
            
            classification = "No cell"
            
            section = image[
                    i_floor:i_ceil,
                    j_floor:j_ceil
            ]
            
            for c in classes:
                if c["nucleus_x"] > i_floor and c["nucleus_x"] < i_ceil and c["nucleus_y"] > j_floor and c["nucleus_y"] < j_ceil:
                    logger.debug(
                        """Cell {}""".format(
                            c
                        )
                    )
                    classification = c["bethesda_system"]
                    break
            
            sections.append(
                (
                    section,
                    classification
                )
            )
            
            logger.debug(
                """Section is {}\n"""
                """\ti_floor: {}\n"""
                """\ti_ceil: {}\n"""
                """\tj_floor: {}\n"""
                """\tj_ceil: {}""".format(
                    classification,
                    i_floor,
                    i_ceil,
                    j_floor,
                    j_ceil
                )
            )
            j = j + 1
        j = 0
        i = i + 1
        
    return sections