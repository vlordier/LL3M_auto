import React, { Suspense, useState, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  ContactShadows,
  useGLTF,
  Html,
  Text,
  Box,
  Sphere,
} from "@react-three/drei";
import { motion } from "framer-motion";
import * as THREE from "three";
import {
  CubeIcon,
  EyeIcon,
  ArrowPathIcon,
  SunIcon,
  Square3Stack3DIcon,
} from "@heroicons/react/24/outline";
import type { ThreeViewerProps } from "@/types";
import { getAssetUrl } from "@/utils";

// Asset loader component
function AssetModel({ url, assetType }: { url: string; assetType: string }) {
  try {
    if (assetType === "gltf") {
      const { scene } = useGLTF(url);
      return <primitive object={scene.clone()} />;
    }

    // For OBJ files, we'd need a different loader
    // For now, show a placeholder
    return <PlaceholderModel />;
  } catch (error) {
    console.error("Failed to load 3D model:", error);
    return <ErrorModel />;
  }
}

// Placeholder model when no asset is loaded
function PlaceholderModel() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
    }
  });

  return (
    <group>
      <Box ref={meshRef} args={[1, 1, 1]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="#3b82f6"
          wireframe
          transparent
          opacity={0.8}
        />
      </Box>
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.2}
        color="#6b7280"
        anchorX="center"
        anchorY="middle"
      >
        No asset loaded
      </Text>
    </group>
  );
}

// Error model when loading fails
function ErrorModel() {
  return (
    <group>
      <Sphere args={[0.5]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#ef4444" transparent opacity={0.7} />
      </Sphere>
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.2}
        color="#dc2626"
        anchorX="center"
        anchorY="middle"
      >
        Failed to load
      </Text>
    </group>
  );
}

// Loading component
function LoadingSpinner() {
  return (
    <Html center>
      <div className="flex items-center gap-3 p-4 bg-white rounded-lg shadow-lg">
        <div className="animate-spin w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full" />
        <span className="text-gray-700">Loading 3D model...</span>
      </div>
    </Html>
  );
}

export function ThreeViewer({
  assetUrl,
  assetType,
  showControls = true,
  showWireframe = false,
  autoRotate = false,
  onLoad,
  onError,
}: ThreeViewerProps) {
  const [viewMode, setViewMode] = useState<"solid" | "wireframe" | "both">(
    "solid",
  );
  const [showGrid, setShowGrid] = useState(false);
  const [environmentPreset, setEnvironmentPreset] = useState<
    "city" | "forest" | "studio"
  >("studio");

  const fullAssetUrl = assetUrl ? getAssetUrl(assetUrl) : undefined;

  const handleReset = () => {
    // Reset camera position - this would need to be implemented via ref
    console.log("Reset camera position");
  };

  return (
    <div
      className="relative w-full h-full bg-gradient-to-br from-gray-100 to-gray-200 rounded-lg overflow-hidden"
      data-testid="three-viewer"
    >
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [5, 5, 5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        shadows
        className="w-full h-full"
      >
        <Suspense fallback={<LoadingSpinner />}>
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={1}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />

          {/* Environment */}
          <Environment preset={environmentPreset} background={false} />

          {/* Ground */}
          <ContactShadows
            position={[0, -1, 0]}
            opacity={0.4}
            scale={20}
            blur={1}
            far={10}
            resolution={512}
            color="#000000"
          />

          {/* Grid */}
          {showGrid && (
            <gridHelper
              args={[10, 10, "#cccccc", "#eeeeee"]}
              position={[0, -1, 0]}
            />
          )}

          {/* 3D Model or Placeholder */}
          {fullAssetUrl && assetType ? (
            <AssetModel url={fullAssetUrl} assetType={assetType} />
          ) : (
            <PlaceholderModel />
          )}

          {/* Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            autoRotate={autoRotate}
            autoRotateSpeed={1}
            maxPolarAngle={Math.PI / 2}
            minDistance={2}
            maxDistance={20}
          />
        </Suspense>
      </Canvas>

      {/* Overlay Controls */}
      {showControls && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-4 left-4 right-4"
        >
          <div className="bg-white/90 backdrop-blur-sm rounded-lg border border-gray-200 shadow-lg">
            <div className="p-3">
              {/* Main controls row */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() =>
                      setViewMode(viewMode === "solid" ? "wireframe" : "solid")
                    }
                    className={`p-2 rounded-lg transition-colors ${
                      viewMode === "wireframe"
                        ? "bg-primary-100 text-primary-700"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                    title="Toggle wireframe"
                  >
                    <Square3Stack3DIcon className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => setShowGrid(!showGrid)}
                    className={`p-2 rounded-lg transition-colors ${
                      showGrid
                        ? "bg-primary-100 text-primary-700"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}
                    title="Toggle grid"
                  >
                    <CubeIcon className="w-4 h-4" />
                  </button>

                  <button
                    onClick={handleReset}
                    className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                    title="Reset camera"
                  >
                    <ArrowPathIcon className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <select
                    value={environmentPreset}
                    onChange={(e) =>
                      setEnvironmentPreset(e.target.value as any)
                    }
                    className="text-sm border border-gray-300 rounded-md px-2 py-1"
                  >
                    <option value="studio">Studio</option>
                    <option value="city">City</option>
                    <option value="forest">Forest</option>
                  </select>

                  <SunIcon className="w-4 h-4 text-gray-500" />
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center gap-4">
                  <span>Use mouse to orbit, zoom, and pan</span>
                  {assetUrl && (
                    <span className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      Asset loaded
                    </span>
                  )}
                </div>

                {assetType && (
                  <span className="uppercase font-mono bg-gray-100 px-2 py-0.5 rounded">
                    {assetType}
                  </span>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Loading overlay */}
      {!assetUrl && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100/50">
          <div className="text-center">
            <CubeIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-600 font-medium">No asset loaded</p>
            <p className="text-gray-500 text-sm">
              Generate a 3D asset to see it here
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// Preload GLTF files for better performance
useGLTF.preload = (url: string) => {
  useGLTF(url);
};
