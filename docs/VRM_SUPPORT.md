# VRM Support in LL3M

LL3M now supports importing VRM (Virtual Reality Model) files - the standard format for 3D avatars used in VR applications, VTubing, and metaverse platforms.

## What is VRM?

VRM is an open source 3D avatar file format based on glTF 2.0. It's specifically designed for VR applications and includes:

- Humanoid avatar specifications
- Materials and textures
- Spring bone physics
- Facial expressions
- Eye tracking support
- Pose constraints

## Installation

To enable VRM support in LL3M, you need to install the VRM-Addon-for-Blender:

### Step 1: Download the VRM Addon

1. Visit the official repository: https://github.com/saturday06/VRM-Addon-for-Blender
2. Download the latest release for your Blender version
3. Download the `.zip` file (do not extract it)

### Step 2: Install in Blender

1. Open Blender
2. Go to `Edit` → `Preferences` → `Add-ons`
3. Click `Install...` button
4. Select the downloaded `.zip` file
5. Enable the addon by checking the checkbox next to "Import-Export: VRM format"
6. Click `Save Preferences`

### Step 3: Verify Installation

You can verify the installation by:
1. Going to `File` → `Import` in Blender
2. Check if "VRM (.vrm)" appears in the import menu

## Usage

Once the VRM addon is installed, LL3M will automatically:

1. **Detect VRM files**: The asset importer will recognize `.vrm` files
2. **Use VRM import**: When importing VRM files, it will use the proper VRM importer
3. **Fallback support**: If the VRM addon is not available, it will try to import as GLB (since VRM is based on glTF)

### Importing VRM Files

You can import VRM files using any of the three import methods:

1. **File Upload**: Drag and drop VRM files or click to browse
2. **File Path**: Provide the absolute path to a VRM file on your system
3. **URL Import**: Download VRM files directly from URLs

### Supported Features

- ✅ VRM file format recognition
- ✅ Avatar mesh import
- ✅ Material and texture import
- ✅ Bone structure preservation
- ✅ Automatic preview generation
- ✅ Export to other formats (GLB, OBJ, FBX, etc.)

## Troubleshooting

### VRM Addon Not Found

If you see "VRM addon not available, trying GLB import as fallback..." in the logs:

1. Make sure you've installed the VRM-Addon-for-Blender correctly
2. Restart Blender after installation
3. Check that the addon is enabled in Preferences → Add-ons
4. Ensure you're using a compatible version of the addon for your Blender version

### Import Errors

If VRM import fails:

1. Check the VRM file is not corrupted
2. Ensure the file follows VRM specification
3. Try importing the file directly in Blender to isolate issues
4. Check Blender console for detailed error messages

### Performance Issues

For large VRM files:

1. Consider reducing texture resolution before import
2. Use lower polygon count versions for previews
3. Monitor system memory usage during import

## Technical Details

### VRM Import Process

1. **Detection**: LL3M checks if the VRM addon is installed using `addon_utils.check("io_scene_vrm")`
2. **Import**: Uses `bpy.ops.import_scene.vrm()` to import the file
3. **Fallback**: If VRM addon unavailable, uses `bpy.ops.import_scene.gltf()` as fallback
4. **Preview**: Generates automatic preview images with proper lighting and camera setup

### File Format Compatibility

- **Primary**: `.vrm` files with VRM-Addon-for-Blender
- **Fallback**: `.vrm` files imported as GLB/glTF 2.0
- **Export**: Can export VRM-imported models to GLTF, GLB, OBJ, FBX, STL formats

## Resources

- [VRM Specification](https://vrm.dev/en/)
- [VRM-Addon-for-Blender GitHub](https://github.com/saturday06/VRM-Addon-for-Blender)
- [VRM Consortium](https://vrm-consortium.org/en/)
- [Blender VRM Documentation](https://vrm-addon-for-blender.info/en/)

## Contributing

If you encounter issues with VRM support or have suggestions for improvements, please:

1. Check if it's a VRM addon issue or LL3M integration issue
2. Report VRM addon bugs to the [upstream repository](https://github.com/saturday06/VRM-Addon-for-Blender/issues)
3. Report LL3M integration issues in this repository

## License

VRM format is open source under MIT license. The VRM-Addon-for-Blender is also open source. Please check individual licenses for compliance.