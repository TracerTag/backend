import svgwrite


def export_to_svg(detected_objects):
    # Create the base SVG to output
    dwg = svgwrite.Drawing("output.svg", profile='tiny')

    # Loop over each detected object
    for detected_object in detected_objects:
        # Create the corresponding path
        object_svg_path = detected_object.to_svg_path()

        # Create the PATH element
        object_svg_element_path = dwg.path(d=object_svg_path, fill='none', stroke='black', stroke_width=1)

        # Set the class as description
        object_svg_element_path.set_desc(desc=detected_object.object_class)

        # Add it to the SVG we are currently generating
        dwg.add(object_svg_element_path)
    dwg.save()
    return dwg
