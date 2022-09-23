import itertools
import cv2
import numpy as np
import math



def shiTomasi(img, feature_params):
    return cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), **feature_params)


def opticalFlow(prevImg, nextImg, prevPts, lk_params=None):
	prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
	nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
	flow, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, None, **lk_params)

	good_old = prevPts[status == 1].astype(int)
	good_new = flow[status == 1].astype(int)

	return good_old, good_new


def cluster_xmeans(data_points, max_clusters=100):
    """Clusters data points into an unspecified number of clusters.

    The number of clusters is arrived at via Bayesian Information Criterion (BIC),
    which privileges the amount of variance explained by a clustering, while penalizing
    the number of clusters. We start with one cluster and split it
    until BIC stops improving. Uses kmeans for clustering.

    Args:
        data_points: List of data points.
        max_clusters: Integer, a hard limit on the maximum number of clusters.
            Can be None for no limit.

    Returns:
        Array of labels for each data point.
    """
    num_clusters = 1
    if max_clusters is None:
        max_clusters = math.inf

    best_labels, best_centers = cluster_kmeans(data_points, num_clusters)
    best_clustering_score = _score_clustering(data_points, best_labels, best_centers)
    while num_clusters < len(data_points) and num_clusters <= max_clusters:
        num_clusters += 1
        labels, centers = cluster_kmeans(data_points, num_clusters)
        clustering_score = _score_clustering(data_points, labels, centers)
        if clustering_score < best_clustering_score:
            best_labels = labels
            best_clustering_score = clustering_score
        else:
            # Results are no longer improving with additional clusters.
            break
    return best_labels


def cluster_kmeans(data_points, num_clusters, max_iterations=100, max_accuracy=0.25):
    """Clusters data points into a given number of clusters.

     Uses kmeans.

    Args:
        data_points: List of data points.
        num_clusters: Integer.
        max_iterations: Integer, how many iterations kmeans is allowed to run.
        max_accuracy: Float, stop kmeans when the impact of a marginal iteration
            falls below this threshold.

    Returns:
        Array of labels for each data point, array of centers for each cluster.
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        max_iterations,
        max_accuracy)
    _, labels, centers = cv2.kmeans(
        data=data_points, K=num_clusters, bestLabels=None, criteria=criteria,
        attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS)
    return [l[0] for l in labels], centers


def _score_clustering(data_points, labels, centers):
    """Calculates a score for how efficiently a given clustering models the data.

    The score is trying to weigh model fit against model complexity.

    Args:
        data_points: List of data points.
        labels: List of integer data point labels, in corresponding order
            to the data_points argument.
        centers: List of cluster centers.

    Returns:
        float, score, lower is better.
    """
    num_obs = data_points.shape[0]
    num_clusters = len(centers)
    total_sse = _get_total_cluster_sse(data_points, labels, centers)
    if total_sse <= 0:
        # A set of duplicate points will trigger this.
        return 0

    # Bayes Information Criterion
    return num_obs * math.log(total_sse / num_obs) + num_clusters * math.log(num_obs)


def _get_total_cluster_sse(data_points, labels, centers):
    """Calculates sum of all within-cluster sum of squared errors.

    Args:
        data_points: List of data points.
        labels: List of integer data point labels, in corresponding order
            to the data_points argument.
        centers: List of cluster centers.

    Returns:
        float, sum of squared error.
    """
    center_to_group_data_points = {}
    for i, point in enumerate(data_points):
        group_center = tuple(centers[labels[i]])
        center_to_group_data_points.setdefault(group_center, []).append(point)
    within_group_sse = 0
    for center, group_points in center_to_group_data_points.items():
        within_group_sse += _get_cluster_sse(group_points, center)
    return within_group_sse


def _get_cluster_sse(data_points, center):
    """Calculates sum of squared error in a single cluster.

    Args:
        data_points: List of data points.
        center: Cluster center.

    Returns:
        float, sum of squared error.
    """
    variance = 0
    for point in data_points:
        variance += _get_distance_between_points(
            tuple(center), tuple(point)) ** 2
    return variance


def _get_distance_between_points(point_a, point_b):
    """Gets distance between two points of any dimensionality.

    Args:
        point_a: Tuple of any size.
        point_b: Tuple of any size.

    Returns:
        float, distance.
    """
    num_dimensions = len(point_a)
    return math.sqrt(sum([abs(point_a[i] - point_b[i]) ** 2
                          for i in range(num_dimensions)]))



def find_all_intersections(lines):
    """Finds intersection points, if any, between all pairs of lines.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2), i.e. two points on the line.

    Returns:
        List of (x, y) intersection points. These are not unique.
    """
    intersection_points = []
    for line_a, line_b in itertools.combinations(lines, 2):
        point = find_intersection(line_a, line_b)
        if point is not None:
            intersection_points.append(point)
    return intersection_points


def find_intersection(line_a, line_b):
    """Finds intersection point between two lines, if any.

    Args:
        line_a: Vector of x1, y1, x2, y2, i.e. two points on the line.
        line_b: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Vector of x, y, the point of intersection. May be None for parallel
        or duplicate lines.
    """
    ax1, ay1, ax2, ay2 = np.array(line_a, dtype=np.float64)
    bx1, by1, bx2, by2 = np.array(line_b, dtype=np.float64)
    denominator = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)
    if denominator == 0:
        return None
    x0 = ((ax1 * ay2 - ay1 * ax2) * (bx1 - bx2) - (
            ax1 - ax2) * (bx1 * by2 - by1 * bx2)) / denominator
    y0 = ((ax1 * ay2 - ay1 * ax2) * (by1 - by2) - (
            ay1 - ay2) * (bx1 * by2 - by1 * bx2)) / denominator
    return x0, y0


def point_to_line_dist(point, line):
    """Finds euclidean distance between a point and a line.

    Args:
        point: Tuple (x, y) point.
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, distance.
    """
    px, py = np.array(point, dtype=np.float64)
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    nominator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    return nominator / point_to_point_dist((x1, y1), (x2, y2))


def point_to_point_dist(point_a, point_b):
    """Finds euclidean distance between two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Float, distance.
    """
    x1, y1 = np.array(point_a, dtype=np.float64)
    x2, y2 = np.array(point_b, dtype=np.float64)
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def find_nearest_point(point_a, points):
    """Finds the point among a set of points nearest to a given point.

    Args:
        point_a: (x, y) point tuple.
        points: Iterable of (x, y) point tuples.

    Returns:
        Tuple of:
            An (x, y) point tuple.
            Its distance to point_a (float).
    """
    nearest_point = None
    nearest_distance = None
    for point in points:
        dist = point_to_point_dist(point_a, point)
        if nearest_point is None or dist < nearest_distance:
            nearest_distance = dist
            nearest_point = point
    return nearest_point, nearest_distance


def get_midpoint(point_a, point_b):
    """Finds the midpoint of two points.

    Args:
        point_a: Tuple (x, y) point.
        point_b: Tuple (x, y) point.

    Returns:
        Tuple of (x, y) midpoint.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_line_angle(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return radians

def get_line_angle_degree(line):
    """Calculates the angle of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, angle in degrees.
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    radians = np.arctan2(y2 - y1, x2 - x1)
    return math.degrees(radians) % 360


def get_line_slope(line):
    """Calculates the slope of a line.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, slope value.

    Raises:
        ZeroDivisionError, if slope is undefined (vertical lines).
    """
    x1, y1, x2, y2 = np.array(line, dtype=np.float64)
    denom = x2 - x1
    if denom == 0:
        raise ZeroDivisionError
    return (y2 - y1) / denom


def get_line_length(line):
    """Calculates the length of a line segment.

    Args:
        line: Vector of x1, y1, x2, y2, i.e. two points on the line.

    Returns:
        Float, line length.
    """
    return point_to_point_dist((line[0], line[1]), (line[2], line[3]))


def find_bounding_points(points):
    """Finds the minimum points that bound the set of points.

    In 2d, this is the lower left and upper right corners of
    a rectangle that contains all the given points. Overlap is
    allowed.

    Args:
        points: List of points.

    Returns:
        Tuple of min and max bounding points tuples.
    """
    if not points:
        return None, None
    min_bounding_point = list(points[0])
    max_bounding_point = list(points[0])
    for point in points:
        for i, coord in enumerate(point):
            min_bounding_point[i] = min(min_bounding_point[i], coord)
            max_bounding_point[i] = max(max_bounding_point[i], coord)
    return min_bounding_point, max_bounding_point


def find_point_on_rect_border(rect, angle):
    """Finds the point on the border at the given angle from center.

    Args:
        rect: Tuple of two points, the lower left and upper right
            corners of the rectangle.
        angle: Angle, in degrees.

    Returns:
        A point tuple on the rectangle's border.
    """
    # Grow the rectangle into a square, and find the border point on that,
    # since it's easier.
    width = abs(rect[1][0] - rect[0][0])
    height = abs(rect[1][1] - rect[0][1])
    square_size = max(width, height)
    square = [rect[0], (rect[0][0] + square_size, rect[0][1] + square_size)]
    border_x, border_y = _find_point_on_square_border(square, angle)
    # Scale the result to find the corresponding point on the original rectangle.
    border_x = border_x * (width / square_size)
    border_y = border_y * (height / square_size)
    return int(round(border_x)), int(round(border_y))


def _find_point_on_square_border(square, angle):
    """Finds the point on the border at the given angle from center.

    Based on: https://stackoverflow.com/a/1343531

    Args:
        square: Tuple of two points, the lower left and upper right
            corners of the square.
        angle: Angle, in degrees.

    Returns:
        A point tuple on the square's border.
    """
    angle = math.radians(angle)
    width = abs(square[1][0] - square[0][0])
    height = abs(square[1][1] - square[0][1])
    assert width == height
    center_x = square[0][0] + width / 2
    center_y = square[0][1] + height / 2
    abs_cos_angle = abs(math.cos(angle))
    abs_sin_angle = abs(math.sin(angle))
    if width / 2 * abs_sin_angle <= height / 2 * abs_cos_angle:
        magnitude = width / 2 / abs_cos_angle
    else:
        magnitude = height / 2 / abs_sin_angle
    border_x = center_x + math.cos(angle) * magnitude
    border_y = center_y + math.sin(angle) * magnitude
    return int(round(border_x)), int(round(border_y))


def find_point_cluster_average(points):
    """Finds the average of a set of points.

    This is a center of mass style average.

    Args:
        points: List of point tuples.

    Returns:
        Point tuple.
    """
    if len(points) == 0:
        return None
    x_sum = np.float64(0)
    y_sum = np.float64(0)
    for x, y in points:
        x_sum += x
        y_sum += y
    return x_sum / len(points), y_sum / len(points)


def get_biggest_intersection(lines, intersection_threshold=3):
    """Finds point with the most lines intersecting.

    Intersections can be considered loosely (within a radius) depending
    on the threshold parameter.

    Args:
        lines: Iterable of lines tuples in (x1, y1, x2, y2) format.
        intersection_threshold: Maximum distance between a line and
            an intersection point for it to be considered a member of
            that intersection.

    Returns:
        Tuple of intersection point, array of lines intersecting there.
        May return (None, None) if no intersections are found.
    """

    intersection_to_lines = _group_lines_by_intersections(
        list(lines), intersection_threshold=intersection_threshold)
    if len(intersection_to_lines) == 0:
        return None, None
    intersection_point, intersection_lines = sorted(
        intersection_to_lines.items(), key=lambda t: len(t[1]))[-1]
    return intersection_point, intersection_lines


def _group_lines_by_intersections(lines, intersection_threshold=3):
    """Group lines according to their intersection points, within some tolerance.

    Args:
        lines: Iterable of lines tuples in (x1, y1, x2, y2) format.
        intersection_threshold: Maximum distance between a line and
            an intersection point for it to be considered a member of
            that intersection.

    Returns:
        Dict of intersection point to set of lines considered to intersect there.
        A line may belong to 0-n intersection points.
    """
    intersection_points = list(set(find_all_intersections(lines)))
    intersection_to_lines = {}
    for point in intersection_points:
        intersection_to_lines[point] = []
        for line in lines:
            if point_to_line_dist(point, line) < intersection_threshold:
                intersection_to_lines[point].append(line)
    return intersection_to_lines


def find_largest_intersection_cluster(lines):
    """Finds the largest cluster of nearby line intersections.

    Args:
        lines: List of lines specified as (x1, y1, x2, y2), i.e. two points on the line.

    Returns:
        List of intersection points, i.e. (x, y) tuples.
    """
    intersection_points = np.float32(np.asarray(find_all_intersections(lines)))
    if len(intersection_points) == 0:
        return []
    elif len(intersection_points) == 1:
        return intersection_points
    labels = clusterer.cluster_xmeans(intersection_points, max_clusters=10)
    label_to_points = {}
    for i, label in enumerate(labels):
        label_to_points.setdefault(label, []).append(intersection_points[i])
    point_clusters = sorted(label_to_points.values(), key=lambda points: len(points))
    return point_clusters[-1]


def enhance_edges(image):
    """Pre-processing step to enhance edges.
    Args:
        image: OpenCV image.
    Returns:
        Image, filtered for edge detection.
    """
    working_image = image.copy()
    working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    # Blur away fine details.
    working_image = cv2.GaussianBlur(working_image, (5, 5), 0)
    return working_image


def lsd_lines(source_image, min_line_length=0.0375, max_line_length=1, min_precision=0):
    """LSD algorithm for line detection.
    Args:
        source_image: An OpenCV Image.
        min_line_length: Minimum line size. Specified as a percentage of the
            source image diagonal (0-1).
        max_line_length: Maximum line size. Specified as a percentage of the
            source image diagonal (0-1).
        min_precision: Minimum precision of detections.
    Returns:
        Array of line endpoints tuples (x1, y1, x2, y2).
    """
    height, width = source_image.shape[:2]
    diagonal = math.sqrt(height ** 2 + width ** 2)
    min_line_length = min_line_length * diagonal
    max_line_length = max_line_length * diagonal
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines, rect_widths, precisions, false_alarms = detector.detect(source_image)
    line_lengths = [get_line_length(l[0]) for l in lines]
    return [l[0] for (i, l) in enumerate(lines)
            if max_line_length > line_lengths[i] > min_line_length and
            precisions[i] > min_precision]


def hough_lines(source_image, min_points=0.075, min_line_length=0.2, max_line_gap=0.2):
    """Hough line detection.
    Args:
        source_image: An OpenCV Image.
        min_points: Float, minimum number of points that must be detected on a line.
            Specified as a percentage of the source image diagonal (0-1).
        min_line_length: Minimum line size. Specified as a percentage of the
            source image diagonal (0-1).
        max_line_gap: Maximum gap between segments of the same line. Specified as
            a percentage of the source image diagonal (0-1).
    Returns:
        Array of line endpoints tuples (x1, y1, x2, y2).
    """
    height, width = source_image.shape[:2]
    size = math.sqrt(height ** 2 + width ** 2)
    lines = cv2.HoughLinesP(
        source_image,
        1,
        np.pi / 180,
        math.ceil(min_points * size),
        minLineLength=math.ceil(min_line_length * size),
        maxLineGap=math.ceil(max_line_gap * size))
    return [l[0] for l in lines] if lines is not None else []


def canny_edges(image, thresholding_sigma=0.33):
    """Detects edges in an image.
    Uses Canny edge detection method. Bases thresholding on
    image statistics combined with the provided sigma.
    Args:
        image: An OpenCV Image.
        thresholding_sigma: Float. Higher values will detect more edges.
    Returns:
        Image with edges marked.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - thresholding_sigma) * v))
    upper = int(min(255, (1.0 + thresholding_sigma) * v))
    edged_image = cv2.Canny(image, lower, upper)
    return edged_image