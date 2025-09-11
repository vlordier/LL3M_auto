# Hierarchical 3D Asset Generation

This system generates 3D assets using hierarchical prompt decomposition, progressing from simple basic geometry to complex animated scenes with CharMorph Vitruvian characters.

## 🎯 Complexity Levels

| Level | Type | Description | Example |
|-------|------|-------------|---------|
| **1** | **Basic Geometry** | Simple shapes, primitives | "Create an ovoid" |
| **2** | **Simple Objects** | Basic furniture, structures | "Generate a wooden table" |
| **3** | **Textured Objects** | Surface materials, textures | "Brick wall with mortar details" |
| **4** | **Lit Scenes** | Environmental lighting | "Cozy living room with warm lighting" |
| **5** | **Complex Animations** | CharMorph characters, motion | "Blonde woman dancing Gangnam Style" |

## 🚀 Quick Start

### Generate Basic Assets (Level 1)
```bash
make generate-basic
```
Generates 3 basic geometry assets (ovoid, cube, sphere) with:
- Python code
- Rendered images
- .blend files
- Exported formats (OBJ, FBX, glTF, PLY, STL)

### Generate All Levels
```bash
make generate-hierarchical
```
Generates 2 assets per level (1-5) showing progression from simple to complex.

### Generate Specific Level
```bash
make generate-level-1  # Basic geometry
make generate-level-2  # Simple objects
make generate-level-3  # Textured objects
make generate-level-4  # Lit scenes
make generate-level-5  # Complex animations (CharMorph Vitruvian)
```

### Ultimate Complexity Test
```bash
make generate-gangnam
```
Generates the most complex test case: **CharMorph Vitruvian blonde woman in bikini dancing Gangnam Style** 💃

## 📁 Output Structure

```
outputs/hierarchical_tests/
├── level_1/
│   ├── level_1_asset_01/
│   │   ├── metadata.json
│   │   ├── code/           # Generated Python scripts
│   │   ├── renders/        # PNG/JPG screenshots
│   │   ├── blends/         # .blend files
│   │   ├── exports/        # OBJ, FBX, glTF, PLY, STL
│   │   └── logs/           # Generation logs
│   └── level_1_asset_02/
├── level_2/
├── level_3/
├── level_4/
├── level_5/
└── generation_results.json
```

## 🎭 CharMorph Vitruvian Integration

Level 5 uses **CharMorph Vitruvian** for realistic human characters:

### Features
- Realistic human proportions
- Facial morphing and expressions
- Body shape adjustment
- Clothing simulation (including bikini)
- Hair systems (blonde, brunette, etc.)
- Animation-ready rigs

### Supported Animations
- **Dance Moves**: Gangnam Style, breakdance, ballet, contemporary
- **Action Sequences**: fighting, climbing, swimming, gymnastics
- **Basic Movements**: walk, run, jump, sit, stand
- **Facial Expressions**: emotional range
- **Lip Sync**: speech animation

## 🛠️ Advanced Usage

### Custom Generation Script
```bash
python scripts/run_hierarchical_generation.py --level 3 --max-per-level 5
```

### Options
- `--level N`: Generate only level N (1-5)
- `--start-level N`: Start from level N
- `--end-level N`: End at level N
- `--max-per-level N`: Limit assets per level
- `--output-dir PATH`: Custom output directory

## 📊 Performance Expectations

| Level | Time | Tasks | Example |
|-------|------|-------|---------|
| 1 | < 30s | 1-3 | Simple ovoid |
| 2 | < 60s | 3-6 | Wooden table |
| 3 | < 120s | 5-10 | Textured brick wall |
| 4 | < 300s | 8-15 | Lit living room |
| 5 | < 600s | 12-25 | Gangnam Style dance |

## 🧪 Testing

### Run Hierarchical Prompt Tests
```bash
pytest tests/integration/test_hierarchical_prompts.py -v
```

### Interactive Demo
```bash
make generate-demo
```

## 🎨 Generated File Types

### Code Files
- **Python scripts**: Blender generation code
- **Export scripts**: Multi-format conversion

### 3D Assets
- **.blend**: Native Blender files
- **.obj**: Wavefront OBJ (universal)
- **.fbx**: Autodesk FBX (animation)
- **.gltf**: glTF 2.0 (web/AR/VR)
- **.ply**: Stanford PLY (research)
- **.stl**: STL (3D printing)

### Renders
- **PNG**: High-quality screenshots
- **JPG**: Compressed previews
- **EXR/HDR**: High dynamic range (Level 4+)

## 🧹 Cleanup

```bash
make clean-assets  # Remove generated assets
make clean         # Full cleanup including assets
```

## 🎯 Use Cases

### Educational
- Teaching 3D modeling progression
- Demonstrating hierarchical decomposition
- Complexity scaling examples

### Testing
- MCTS planner validation
- Quality assessment benchmarks
- Performance profiling

### Production
- Rapid prototyping pipeline
- Asset generation automation
- Quality control testing

---

**Ready to generate?** Start with `make generate-basic` and work your way up to the ultimate `make generate-gangnam`! 🕺💃
