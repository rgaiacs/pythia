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
    
    while (j + 1) * section_size < image.shape[0]:
        j_floor = j * 100
        j_ceil = (j + 1) * 100
        while (i + 1) * section_size < image.shape[1]:
            classification = "No cell"
            i_floor = j * 100
            i_ceil = (j + 1) * 100
            
            section = image[
                    j_floor:j_ceil,
                    i_floor:i_ceil
            ]
            
            for c in classes:
                if c["nucleus_x"] > j_floor and c["nucleus_x"] < j_ceil and c["nucleus_y"] > i_floor and c["nucleus_y"] > i_floor:
                    classification = c["bethesda_system"]
                    break
            
            sections.append(
                (
                    section,
                    classification
                )
            )
            i = i + 1
        i = 0
        j = j + 1
        
    return sections