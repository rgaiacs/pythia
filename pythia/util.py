"""Utils"""
import json
import logging
import os.path

from sklearn.model_selection import KFold

from . import io

from . import hog
from . import glcm
from . import lbp
from . import hu

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
LOGGER = logging.getLogger(__name__)

def sample2features(sample):
    """Generate features for image"""
    features = []
    
    # Local binary pattern
    features.extend(lbp.hist(sample))

    # Histogram of Oriented Gradients
    features.extend(hog.hist(sample))

    # Gray-Level Co-Occurrence Matrix
    features.extend(glcm.hist(sample))

    features.extend(hu.hist(sample))
    
    return features

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
    i = 0
    j = 0
    
    # How the image is sliced into sections.
    # +---+---+---+
    # | 1 | 2 | 3 |
    # +---+---+---+
    # | 4 | 5 | 6 |
    # +---+---+---+
    # | 7 | 8 | 9 |
    # +---+---+---+
    while (i + 1) * section_size < image.shape[0]:
        while (j + 1) * section_size < image.shape[1]:
            yield image[
                i * section_size:(i + 1) * section_size,
                j * section_size:(j + 1) * section_size
            ]

            j = j + 1
        j = 0
        i = i + 1

def image2sections_and_classes(
        image,
        classes,
        section_size=100,
        crop_grid=True,
        crop_center=True,
        binary_classification=True):
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
    if crop_grid:
        # How the image is sliced into sections.
        # +---+---+---+
        # | 1 | 2 | 3 |
        # +---+---+---+
        # | 4 | 5 | 6 |
        # +---+---+---+
        # | 7 | 8 | 9 |
        # +---+---+---+
        i = 0
        j = 0
        while (i + 1) * section_size < image.shape[0]:
            i_floor = i * section_size
            i_ceil = (i + 1) * section_size
            while (j + 1) * section_size < image.shape[1]:
                j_floor = j * section_size
                j_ceil = (j + 1) * section_size

                if binary_classification:
                    classification = "normal cell"
                else:
                    classification = "Negative for intraepithelial lesion"
                
                section = image[
                    i_floor:i_ceil,
                    j_floor:j_ceil
                ]
                
                for cell in classes:
                    if (cell["nucleus_y"] > i_floor and
                            cell["nucleus_y"] < i_ceil and
                            cell["nucleus_x"] > j_floor and
                            cell["nucleus_x"] < j_ceil):
                        LOGGER.debug(
                            "Cell %s",
                            cell
                        )

                        if binary_classification:
                            if cell["bethesda_system"] == "Negative for intraepithelial lesion":
                                classification = "normal cell"
                            else:
                                classification = "altered cell"
                            break
                        else:
                            classification = cell["bethesda_system"]
                            break
                
                LOGGER.debug(
                    """Section is %s\n"""
                    """\ti_floor: %s\n"""
                    """\ti_ceil: %s\n"""
                    """\tj_floor: %s\n"""
                    """\tj_ceil: %s""",
                    classification,
                    i_floor,
                    i_ceil,
                    j_floor,
                    j_ceil
                )

                yield (
                    section,
                    classification
                )
                
                j = j + 1
            j = 0
            i = i + 1

    if crop_center:
        # Sections around classified cells
        half_section_size = int(section_size / 2)
        for cell in classes:
            if binary_classification:
                if cell["bethesda_system"] == "Negative for intraepithelial lesion":
                    classification = "normal cell"
                else:
                    classification = "altered cell"
            else:
                classification = cell["bethesda_system"]

            if cell["nucleus_y"] < half_section_size:
                section_i_start = 0
                section_i_end = section_size
            elif image.shape[0] - cell["nucleus_y"] < half_section_size:
                section_i_start = image.shape[0] - section_size
                section_i_end = image.shape[0]
            else:
                section_i_start = cell["nucleus_y"] - half_section_size
                section_i_end = cell["nucleus_y"] + half_section_size
            if cell["nucleus_x"] < half_section_size:
                section_j_start = 0
                section_j_end = section_size
            elif image.shape[1] - cell["nucleus_x"] < half_section_size:
                section_j_start = image.shape[1] - section_size
                section_j_end = image.shape[1]
            else:
                section_j_start = cell["nucleus_x"] - half_section_size
                section_j_end = cell["nucleus_x"] + half_section_size
            
            LOGGER.debug(
                """Section around cell %s\n"""
                """\tclassification: %s\n"""
                """\ti_floor: %s\n"""
                """\tnucleus_y: %s\n"""
                """\ti_ceil: %s\n"""
                """\tj_floor: %s\n"""
                """\tnucleus_x: %s\n"""
                """\tj_ceil: %s""",
                cell["cell_id"],
                classification,
                section_i_start,
                cell["nucleus_y"],
                section_i_end,
                section_j_start,
                cell["nucleus_x"],
                section_j_end
            )

            section = image[
                section_i_start:section_i_end,
                section_j_start:section_j_end
            ]

            yield (
                section,
                classification
            )

def collection2sections_and_classes(
        data_filename,
        image_folder,
        section_size=100,
        crop_grid=True,
        crop_center=True,
        binary_classification=True):
    """Generate sections and classes for collection"""
    data = io.jsonread(
        data_filename
    )
    
    for datum in data:
        LOGGER.info(
            'Processing image #%s ...\n\tFile name: %s',
            datum["image_id"],
            datum["image_name"]
        )

        image = io.imread2gray(
            os.path.join(
                image_folder,
                datum["image_name"]
            )
        )

        for section, classification in image2sections_and_classes(
                image,
                datum["classifications"],
                section_size,
                crop_grid,
                crop_center,
                binary_classification):
            yield (
                section,
                classification
            )

        LOGGER.info(
            'Finished with image #%s!',
            datum["image_id"]
        )

def collection2features_and_classes(
        data_filename,
        image_folder,
        section_size=100,
        crop_grid=True,
        crop_center=True,
        binary_classification=True):
    """Generate features and classes for collection"""
    features = []
    classifications = []

    for section, classification in collection2sections_and_classes(
            data_filename,
            image_folder,
            section_size,
            crop_grid,
            crop_center,
            binary_classification):
        features.append(
            sample2features(section)
        )
        classifications.append(classification)
            
    return (features, classifications)

def kfold(
        data_filename,
        n_splits=10,
        shuffle=False,
        random_state=None):
    """Provides train/test indices to split data in train/test sets."""
    images = io.jsonread(
        data_filename
    )
    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    set_counter = 0
    for train_index, test_index in kf.split(images):
        set_counter = set_counter + 1
        LOGGER.info(
            'Processing train and test set %s ...',
            set_counter
        )

        train_filename = "train-{}.json".format(set_counter)
        test_filename = "test-{}.json".format(set_counter)

        train = []
        test = []

        for i in train_index:
            train.append(images[i])
        for i in test_index:
            test.append(images[i])

        with open(train_filename, "w") as train_file:
            LOGGER.info(
                'Writing %s ...',
                train_filename
            )
            json.dump(
                train,
                train_file
            )
            LOGGER.info(
                'Fineshed with %s!',
                train_filename
            )

        with open(test_filename, "w") as test_file:
            LOGGER.info(
                'Writing %s ...',
                test_filename
            )
            json.dump(
                test,
                test_file
            )
            LOGGER.info(
                'Fineshed with %s!',
                test_filename
            )

        LOGGER.info(
            'Fineshed with train and test set %s!',
            set_counter
        )
