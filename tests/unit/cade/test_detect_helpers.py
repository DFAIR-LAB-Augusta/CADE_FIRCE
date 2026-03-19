import numpy as np
import pytest

from cade.detect import (
    get_latent_data_for_each_family,
    get_latent_distance_between_sample_and_centroid,
    get_mad_for_each_family,
)


@pytest.mark.unit
@pytest.mark.fast
def test_get_latent_data_for_each_family_groups_vectors() -> None:
    z_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int32)

    n, n_family, z_family = get_latent_data_for_each_family(z_train, y_train)

    assert n == 2
    assert n_family == [2, 2]
    assert np.array_equal(z_family[0], np.array(
        [[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    assert np.array_equal(
        z_family[1],
        np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32),
    )


@pytest.mark.unit
@pytest.mark.fast
def test_get_latent_distance_between_sample_and_centroid() -> None:
    z_family = [
        np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
    ]
    centroids = [
        np.array([0.0, 0.0], dtype=np.float32),
    ]

    distances = get_latent_distance_between_sample_and_centroid(
        z_family=z_family,
        centroids=centroids,
        _margin=10.0,
        n=1,
        n_family=[2],
    )

    assert distances == [[0.0, 5.0]]


@pytest.mark.unit
@pytest.mark.fast
def test_get_mad_for_each_family_computes_expected_values() -> None:
    dis_family = [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ]

    mad = get_mad_for_each_family(dis_family=dis_family, n=2, n_family=[3, 3])

    assert mad[0] == 0.0
    assert pytest.approx(mad[1]) == 1.4826
