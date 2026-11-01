/**
 * AMR Predict — Immersive 3D Spatial Environment Engine
 * Primary Design Reference: George El Dahr (georgedo.com)
 *
 * Visual & Spatial Architecture:
 *  - Scene 01 (Genesis): Guiding bioluminescent energy core in deep atmospheric fog.
 *  - Scene 02 (Microcosm): Smooth camera descent into confocal microscopic landscape.
 *  - Scene 03 (Specimen 3D): Interactive 3D scientific macromolecule with mouse drag orbit,
 *    switchable spectroscopy (Confocal, Dark-Field, Electron), and multi-layer toggles.
 *  - Scene 04 (Intelligence): Camera glide into neural telemetry field.
 *  - Scene 05 (Clinical Nexus): Powerful transition gateway into the operational platform.
 *  - Smooth scroll-controlled camera spline interpolation with inertia.
 *  - Interactive spatial raycasting & hotspot tooltips.
 */

(function () {
    'use strict';

    if (typeof THREE === 'undefined') {
        console.warn('Three.js is required for AMR 3D Engine.');
        return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // WebGL Canvas
    let canvas = document.getElementById('amr-3d-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'amr-3d-canvas';
        document.body.prepend(canvas);
    }

    canvas.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        opacity: 1;
        transition: opacity 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        display: block;
    `;

    // Core Components
    let renderer, scene, camera;
    let clock = new THREE.Clock();
    let animFrameId = null;
    let isLandingActive = true;

    // 3D Objects
    let guidingProbeGroup, guidingLight;
    let specimenGroup, outerLatticeMesh, porinMeshGroup, receptorMeshGroup, coreHelixGroup, peptidoglycanGroup;
    let intelligenceGroup, intelligenceCore, targetNodesGroup, synapticVectorsGroup, inputRaysGroup;
    let bioParticlePoints, bokehPoints;

    // Materials dictionary for spectroscopy switching
    let materials = {};

    // Camera & Scroll State
    let scrollProgress = 0; // 0.0 to 1.0 across the 5 landing scenes
    let currentCamPos = new THREE.Vector3(0, 0, 140);
    let targetCamPos = new THREE.Vector3(0, 0, 140);
    let currentLookAt = new THREE.Vector3(0, 0, 0);
    let targetLookAt = new THREE.Vector3(0, 0, 0);

    // Mouse Parallax & Orbit Drag State
    let mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let isDraggingSpecimen = false;
    let previousMousePosition = { x: 0, y: 0 };
    let specimenRotation = { x: 0.2, y: 0.4 };
    let targetSpecimenRotation = { x: 0.2, y: 0.4 };

    // 5 Spatial Camera Keyframes matching the 5 Scenes
    const sceneKeyframes = [
        // Scene 01: Genesis & Threshold
        {
            camPos: new THREE.Vector3(0, 0, 145),
            lookAt: new THREE.Vector3(0, 0, 0),
            probePos: new THREE.Vector3(0, 0, 40),
            specimenScale: 0.35,
            specimenPos: new THREE.Vector3(0, -60, -80),
            intelPos: new THREE.Vector3(0, -80, -100),
            intelScale: 0.1,
            intelOpacity: 0
        },
        // Scene 02: Microcosm Descent
        {
            camPos: new THREE.Vector3(-32, -16, 105),
            lookAt: new THREE.Vector3(12, -8, -15),
            probePos: new THREE.Vector3(-15, -10, 20),
            specimenScale: 0.75,
            specimenPos: new THREE.Vector3(25, -5, -20),
            intelPos: new THREE.Vector3(-40, -40, -80),
            intelScale: 0.2,
            intelOpacity: 0
        },
        // Scene 03: Interactive 3D Specimen Stage (Centrally Aligned & Substantially Larger)
        {
            camPos: new THREE.Vector3(0, -4.2, 54),
            lookAt: new THREE.Vector3(0, -4.2, 0),
            probePos: new THREE.Vector3(32, 22, 25),
            specimenScale: 1.45,
            specimenPos: new THREE.Vector3(0, -4.2, 0),
            intelPos: new THREE.Vector3(0, -60, -100),
            intelScale: 0.2,
            intelOpacity: 0
        },
        // Scene 04: Scientific Intelligence & Neural Multi-Target Network
        {
            camPos: new THREE.Vector3(0, 0, 72),
            lookAt: new THREE.Vector3(0, 0, 0),
            probePos: new THREE.Vector3(20, 15, 30),
            specimenScale: 0.25,
            specimenPos: new THREE.Vector3(60, 40, -120),
            intelPos: new THREE.Vector3(-24, 0, 0),
            intelScale: 1.18,
            intelOpacity: 1
        },
        // Scene 05: Clinical Nexus & Launch Gateway
        {
            camPos: new THREE.Vector3(0, -8, 155),
            lookAt: new THREE.Vector3(0, 0, 0),
            probePos: new THREE.Vector3(0, -2, 50),
            specimenScale: 0.15,
            specimenPos: new THREE.Vector3(0, -60, -120),
            intelPos: new THREE.Vector3(0, 32, -70),
            intelScale: 0.5,
            intelOpacity: 0.2
        }
    ];

    // Texture Generators
    function createGlowTexture(colorHex = '#00F2C3') {
        const size = 128;
        const c = document.createElement('canvas');
        c.width = size;
        c.height = size;
        const ctx = c.getContext('2d');

        const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
        grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
        grad.addColorStop(0.2, 'rgba(0, 242, 195, 0.85)');
        grad.addColorStop(0.5, 'rgba(0, 210, 255, 0.28)');
        grad.addColorStop(0.8, 'rgba(2, 132, 199, 0.08)');
        grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, size, size);

        const tex = new THREE.CanvasTexture(c);
        tex.needsUpdate = true;
        return tex;
    }

    function createBokehTexture() {
        const size = 128;
        const c = document.createElement('canvas');
        c.width = size;
        c.height = size;
        const ctx = c.getContext('2d');

        const grad = ctx.createRadialGradient(size / 2, size / 2, size * 0.2, size / 2, size / 2, size / 2);
        grad.addColorStop(0.0, 'rgba(0, 210, 255, 0.25)');
        grad.addColorStop(0.6, 'rgba(2, 132, 199, 0.12)');
        grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
        ctx.fill();

        const tex = new THREE.CanvasTexture(c);
        tex.needsUpdate = true;
        return tex;
    }

    function init() {
        // Scene with atmospheric fog
        scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x040711, 0.0028);

        // Perspective Camera
        camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 1, 1000);
        camera.position.copy(currentCamPos);

        // WebGL Renderer
        renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            alpha: true,
            antialias: true,
            powerPreference: 'high-performance'
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;

        // Cinematic Lighting
        setupLighting();

        // 1. Guiding Bioluminescent Probe (Inspired by georgedo.com's guiding light)
        buildGuidingProbe();

        // 2. Interactive 3D Scientific Specimen (Scene 03)
        buildScientificSpecimen();

        // 2B. Machine Learning Intelligence Network (Scene 04)
        buildIntelligenceNetwork();

        // 3. Multi-Plane Cellular Depth Field
        buildCellularDepthField();

        // Event Listeners
        window.addEventListener('resize', onResize, { passive: true });
        window.addEventListener('mousemove', onMouseMove, { passive: true });
        window.addEventListener('scroll', onScroll, { passive: true });
        document.addEventListener('visibilitychange', onVisibilityChange, { passive: true });

        // Specimen drag rotation handlers for Scene 03
        setupSpecimenDragControls();

        // Initial update
        onScroll();

        // Start render loop
        if (!prefersReducedMotion) {
            startLoop();
        }
    }

    function setupLighting() {
        // Soft deep ambient base
        const ambient = new THREE.AmbientLight(0x060c1c, 1.8);
        scene.add(ambient);

        // Key light — laboratory cyan/teal illumination
        const keyLight = new THREE.DirectionalLight(0x00D2FF, 2.2);
        keyLight.position.set(60, 70, 80);
        scene.add(keyLight);

        // Rim light — cool electric blue for specular edge separation
        const rimLight = new THREE.DirectionalLight(0x38BDF8, 1.8);
        rimLight.position.set(-60, -40, -50);
        scene.add(rimLight);
    }

    // ─── 1. Guiding Bioluminescent Probe ─────────────────────────────────────────
    function buildGuidingProbe() {
        guidingProbeGroup = new THREE.Group();

        // Inner glowing core
        const coreGeo = new THREE.SphereGeometry(1.6, 24, 24);
        const coreMat = new THREE.MeshBasicMaterial({
            color: 0x00F2C3,
            transparent: true,
            opacity: 0.95
        });
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        guidingProbeGroup.add(coreMesh);

        // Pulsing outer halo
        const haloGeo = new THREE.SphereGeometry(3.2, 24, 24);
        const haloMat = new THREE.MeshBasicMaterial({
            color: 0x00D2FF,
            transparent: true,
            opacity: 0.32,
            wireframe: true
        });
        const haloMesh = new THREE.Mesh(haloGeo, haloMat);
        guidingProbeGroup.add(haloMesh);

        // Volumetric point light tracking the probe
        guidingLight = new THREE.PointLight(0x00F2C3, 3.2, 180, 1.8);
        guidingProbeGroup.add(guidingLight);

        guidingProbeGroup.position.set(0, 0, 40);
        scene.add(guidingProbeGroup);
    }

        // ─── 2. Interactive 3D Scientific Specimen (Scene 03) ────────────────────────
    function buildScientificSpecimen() {
        specimenGroup = new THREE.Group();

        // High-fidelity Materials for Advanced Spectroscopy
        materials.confocalLattice = new THREE.MeshPhysicalMaterial({
            color: 0x00F2C3,
            emissive: 0x0369A1,
            emissiveIntensity: 0.5,
            roughness: 0.18,
            metalness: 0.25,
            clearcoat: 0.95,
            clearcoatRoughness: 0.1,
            transparent: true,
            opacity: 0.9
        });

        materials.membraneLipid = new THREE.MeshPhysicalMaterial({
            color: 0x0284C7,
            emissive: 0x0369A1,
            emissiveIntensity: 0.35,
            roughness: 0.2,
            transmission: 0.35,
            transparent: true,
            opacity: 0.65,
            depthWrite: false
        });

        materials.confocalWire = new THREE.LineBasicMaterial({
            color: 0x00D2FF,
            transparent: true,
            opacity: 0.7
        });

        materials.porin = new THREE.MeshPhysicalMaterial({
            color: 0x38BDF8,
            emissive: 0x0284C7,
            emissiveIntensity: 0.6,
            roughness: 0.15,
            metalness: 0.5
        });

        materials.porinLumen = new THREE.MeshBasicMaterial({
            color: 0x00F2C3,
            transparent: true,
            opacity: 0.85
        });

        materials.receptor = new THREE.MeshPhysicalMaterial({
            color: 0xF59E0B, // Calibrated amber receptor binding site
            emissive: 0xD97706,
            emissiveIntensity: 0.75,
            roughness: 0.2,
            metalness: 0.4
        });

        materials.peptidoglycan = new THREE.LineBasicMaterial({
            color: 0x6366F1,
            transparent: true,
            opacity: 0.45
        });

        materials.coreHelix = new THREE.MeshPhysicalMaterial({
            color: 0x818CF8,
            emissive: 0x4F46E5,
            emissiveIntensity: 0.6,
            roughness: 0.25,
            metalness: 0.4
        });

        // A. Dual-Layer Outer Membrane Matrix & Capsid Geodesic
        outerLatticeMesh = new THREE.Group();

        // Outer translucent lipid bilayer envelope
        const lipidGeo = new THREE.SphereGeometry(17.5, 36, 36);
        const lipidMesh = new THREE.Mesh(lipidGeo, materials.membraneLipid);
        outerLatticeMesh.add(lipidMesh);

        // Geodesic macromolecular structural lattice
        const icoGeo = new THREE.IcosahedronGeometry(17.8, 2);
        const nodeGeo = new THREE.SphereGeometry(0.55, 12, 12);
        const pos = icoGeo.attributes.position;
        const uniqueVertices = new Map();

        for (let i = 0; i < pos.count; i++) {
            const vx = Math.round(pos.getX(i) * 100) / 100;
            const vy = Math.round(pos.getY(i) * 100) / 100;
            const vz = Math.round(pos.getZ(i) * 100) / 100;
            const key = `${vx},${vy},${vz}`;

            if (!uniqueVertices.has(key)) {
                uniqueVertices.set(key, true);
                const atom = new THREE.Mesh(nodeGeo, materials.confocalLattice);
                atom.position.set(vx, vy, vz);
                outerLatticeMesh.add(atom);
            }
        }

        const wireGeo = new THREE.WireframeGeometry(icoGeo);
        const wireSegments = new THREE.LineSegments(wireGeo, materials.confocalWire);
        outerLatticeMesh.add(wireSegments);
        specimenGroup.add(outerLatticeMesh);

        // B. 12 Transmembrane Porin Channels (OmpF/OmpC/PhoE Homologues)
        porinMeshGroup = new THREE.Group();
        const porinOuterGeo = new THREE.CylinderGeometry(1.2, 1.5, 4.8, 16, 1, true);
        const porinLumenGeo = new THREE.CylinderGeometry(0.65, 0.65, 5.0, 12);

        // Geometrically distributed across the spherical surface
        const phiSpan = [Math.PI / 4, Math.PI / 2, 3 * Math.PI / 4];
        let pCount = 0;
        for (let p = 0; p < phiSpan.length; p++) {
            const phi = phiSpan[p];
            const rings = (p === 1) ? 6 : 3;
            for (let r = 0; r < rings; r++) {
                const theta = (r / rings) * Math.PI * 2;
                const rad = 17.5;
                const x = rad * Math.sin(phi) * Math.cos(theta);
                const y = rad * Math.cos(phi);
                const z = rad * Math.sin(phi) * Math.sin(theta);

                const porinUnit = new THREE.Group();
                const barrel = new THREE.Mesh(porinOuterGeo, materials.porin);
                const lumen = new THREE.Mesh(porinLumenGeo, materials.porinLumen);
                porinUnit.add(barrel);
                porinUnit.add(lumen);

                porinUnit.position.set(x, y, z);
                porinUnit.lookAt(0, 0, 0);
                porinMeshGroup.add(porinUnit);
                pCount++;
            }
        }
        specimenGroup.add(porinMeshGroup);

        // C. Target Receptor Complexes & Docking Rings (Penicillin-Binding Proteins / Receptors)
        receptorMeshGroup = new THREE.Group();
        const ringGeo = new THREE.TorusGeometry(25.5, 0.35, 12, 90);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x00F2C3, transparent: true, opacity: 0.35 });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.rotation.x = Math.PI / 3.2;
        ringMesh.rotation.y = Math.PI / 6;
        receptorMeshGroup.add(ringMesh);

        const recGeo = new THREE.DodecahedronGeometry(1.1);
        for (let j = 0; j < 6; j++) {
            const th = (j / 6) * Math.PI * 2;
            const rec = new THREE.Mesh(recGeo, materials.receptor);
            rec.position.set(Math.cos(th) * 25.5, Math.sin(th) * 25.5, 0);
            ringMesh.add(rec);
        }
        specimenGroup.add(receptorMeshGroup);

        // D. Inner Peptidoglycan Cell Wall Mesh
        peptidoglycanGroup = new THREE.Group();
        const pepGeo = new THREE.IcosahedronGeometry(11.5, 1);
        const pepWire = new THREE.WireframeGeometry(pepGeo);
        const pepMesh = new THREE.LineSegments(pepWire, materials.peptidoglycan);
        peptidoglycanGroup.add(pepMesh);
        specimenGroup.add(peptidoglycanGroup);

        // E. Supercoiled Central Genomic Core
        coreHelixGroup = new THREE.Group();
        const coreGeo = new THREE.TorusKnotGeometry(6.5, 1.35, 128, 16, 3, 5);
        const coreKnot = new THREE.Mesh(coreGeo, materials.coreHelix);
        coreHelixGroup.add(coreKnot);
        specimenGroup.add(coreHelixGroup);

        specimenGroup.position.set(0, -60, -80);
        specimenGroup.scale.setScalar(0.35);
        scene.add(specimenGroup);
    }

    // ─── 2B. Multi-Target Machine Learning Intelligence Network (Scene 04) ──────
    function buildIntelligenceNetwork() {
        intelligenceGroup = new THREE.Group();

        // 1. Central Calibrated Tournament Engine Core
        intelligenceCore = new THREE.Group();

        // Outer calibrated ensemble dodecahedron cage
        const coreCageGeo = new THREE.DodecahedronGeometry(6.2, 1);
        const coreWireGeo = new THREE.WireframeGeometry(coreCageGeo);
        const coreWireMat = new THREE.LineBasicMaterial({
            color: 0x00D2FF,
            transparent: true,
            opacity: 0.65
        });
        const coreCage = new THREE.LineSegments(coreWireGeo, coreWireMat);
        intelligenceCore.add(coreCage);

        // Inner probability density octahedron nucleus
        const nucGeo = new THREE.OctahedronGeometry(4.2, 0);
        const nucMat = new THREE.MeshPhysicalMaterial({
            color: 0x00F2C3,
            emissive: 0x0284C7,
            emissiveIntensity: 0.75,
            roughness: 0.15,
            metalness: 0.35,
            transparent: true,
            opacity: 0.85
        });
        const nucMesh = new THREE.Mesh(nucGeo, nucMat);
        intelligenceCore.add(nucMesh);

        // Central bioluminescent energy sphere
        const energyGeo = new THREE.SphereGeometry(1.8, 16, 16);
        const energyMat = new THREE.MeshBasicMaterial({
            color: 0xFFFFFF,
            transparent: true,
            opacity: 0.95
        });
        const energySphere = new THREE.Mesh(energyGeo, energyMat);
        intelligenceCore.add(energySphere);

        intelligenceGroup.add(intelligenceCore);

        // 2. 8 Antibiotic Target Nodes in Orbital Constellation
        targetNodesGroup = new THREE.Group();
        synapticVectorsGroup = new THREE.Group();

        const targets = [
            { code: 'CIP', name: 'Ciprofloxacin', color: 0x38BDF8, angle: 0 },
            { code: 'LVX', name: 'Levofloxacin', color: 0x00D2FF, angle: Math.PI / 4 },
            { code: 'CRO', name: 'Ceftriaxone', color: 0x00F2C3, angle: Math.PI / 2 },
            { code: 'AMC', name: 'Amoxicillin-Clav', color: 0x10B981, angle: 3 * Math.PI / 4 },
            { code: 'NIT', name: 'Nitrofurantoin', color: 0xF59E0B, angle: Math.PI },
            { code: 'SXT', name: 'Trimethoprim-Sulfa', color: 0x818CF8, angle: 5 * Math.PI / 4 },
            { code: 'FOS', name: 'Fosfomycin', color: 0x06B6D4, angle: 3 * Math.PI / 2 },
            { code: 'GEN', name: 'Gentamicin', color: 0x6366F1, angle: 7 * Math.PI / 4 }
        ];

        const nodeRadius = 18.5;
        const targetNodeGeo = new THREE.SphereGeometry(1.25, 16, 16);
        const haloRingGeo = new THREE.TorusGeometry(2.1, 0.08, 8, 32);

        targets.forEach((tgt, idx) => {
            const tx = Math.cos(tgt.angle) * nodeRadius;
            const ty = Math.sin(tgt.angle) * (nodeRadius * 0.75); // subtle elliptical tilt
            const tz = Math.sin(tgt.angle * 2) * 3.5;

            // Target Node Group
            const tGroup = new THREE.Group();
            tGroup.position.set(tx, ty, tz);

            const tMat = new THREE.MeshPhysicalMaterial({
                color: tgt.color,
                emissive: tgt.color,
                emissiveIntensity: 0.65,
                roughness: 0.2,
                metalness: 0.3
            });
            const tSphere = new THREE.Mesh(targetNodeGeo, tMat);
            tGroup.add(tSphere);

            const ringMat = new THREE.MeshBasicMaterial({
                color: tgt.color,
                transparent: true,
                opacity: 0.5
            });
            const ring = new THREE.Mesh(haloRingGeo, ringMat);
            tGroup.add(ring);

            targetNodesGroup.add(tGroup);

            // Flowing Synaptic Vector Spline (from Core to Target Node)
            const curve = new THREE.QuadraticBezierCurve3(
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(tx * 0.5, ty * 0.5, tz + 4.5),
                new THREE.Vector3(tx, ty, tz)
            );

            const points = curve.getPoints(28);
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({
                color: tgt.color,
                transparent: true,
                opacity: 0.45
            });
            const splineLine = new THREE.Line(lineGeo, lineMat);
            synapticVectorsGroup.add(splineLine);
        });

        intelligenceGroup.add(targetNodesGroup);
        intelligenceGroup.add(synapticVectorsGroup);

        // 3. 16 Input Vector Streams (Incoming Clinical Variables)
        inputRaysGroup = new THREE.Group();
        const inputCount = 12;
        for (let k = 0; k < inputCount; k++) {
            const inAngle = (k / inputCount) * Math.PI - (Math.PI / 2);
            const startX = -32;
            const startY = inAngle * 14;
            const startZ = (Math.random() - 0.5) * 12;

            const inCurve = new THREE.QuadraticBezierCurve3(
                new THREE.Vector3(startX, startY, startZ),
                new THREE.Vector3(-14, startY * 0.4, startZ * 0.5),
                new THREE.Vector3(0, 0, 0)
            );
            const inPoints = inCurve.getPoints(20);
            const inGeo = new THREE.BufferGeometry().setFromPoints(inPoints);
            const inMat = new THREE.LineBasicMaterial({
                color: 0x38BDF8,
                transparent: true,
                opacity: 0.28
            });
            inputRaysGroup.add(new THREE.Line(inGeo, inMat));
        }
        intelligenceGroup.add(inputRaysGroup);

        intelligenceGroup.position.set(0, -80, -100);
        intelligenceGroup.scale.setScalar(0.1);
        intelligenceGroup.visible = false;
        scene.add(intelligenceGroup);
    }

    // ─── 3. Multi-Plane Cellular Depth Field ─────────────────────────────────────
    function buildCellularDepthField() {
        const glowTex = createGlowTexture();
        const bokehTex = createBokehTexture();

        // 340 In-focus Cellular Fluorophores
        const inFocusCount = 380;
        const pos = new Float32Array(inFocusCount * 3);
        const col = new Float32Array(inFocusCount * 3);
        const palette = [
            new THREE.Color(0x00F2C3),
            new THREE.Color(0x00D2FF),
            new THREE.Color(0x38BDF8),
            new THREE.Color(0x818CF8)
        ];

        for (let i = 0; i < inFocusCount; i++) {
            const i3 = i * 3;
            pos[i3] = (Math.random() - 0.5) * 340;
            pos[i3 + 1] = (Math.random() - 0.5) * 220;
            pos[i3 + 2] = (Math.random() - 0.5) * 200;

            const c = palette[Math.floor(Math.random() * palette.length)];
            col[i3] = c.r;
            col[i3 + 1] = c.g;
            col[i3 + 2] = c.b;
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

        const mat = new THREE.PointsMaterial({
            size: 3.2,
            vertexColors: true,
            map: glowTex,
            transparent: true,
            opacity: 0.82,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            sizeAttenuation: true
        });

        bioParticlePoints = new THREE.Points(geo, mat);
        scene.add(bioParticlePoints);

        // 40 Deep Bokeh / Peripheral Discs
        const bokehCount = 45;
        const bPos = new Float32Array(bokehCount * 3);
        const bCol = new Float32Array(bokehCount * 3);

        for (let j = 0; j < bokehCount; j++) {
            const j3 = j * 3;
            const isNear = Math.random() > 0.65;
            bPos[j3] = (Math.random() - 0.5) * 360;
            bPos[j3 + 1] = (Math.random() - 0.5) * 240;
            bPos[j3 + 2] = isNear ? (30 + Math.random() * 40) : (-100 - Math.random() * 120);

            const c = isNear ? new THREE.Color(0x00D2FF) : new THREE.Color(0x0369A1);
            bCol[j3] = c.r;
            bCol[j3 + 1] = c.g;
            bCol[j3 + 2] = c.b;
        }

        const bGeo = new THREE.BufferGeometry();
        bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
        bGeo.setAttribute('color', new THREE.BufferAttribute(bCol, 3));

        const bMat = new THREE.PointsMaterial({
            size: 26.0,
            vertexColors: true,
            map: bokehTex,
            transparent: true,
            opacity: 0.35,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            sizeAttenuation: true
        });

        bokehPoints = new THREE.Points(bGeo, bMat);
        scene.add(bokehPoints);
    }

    // ─── Scroll & Camera Choreography ──────────────────────────────────────────
    function onScroll() {
        if (!isLandingActive) return;

        const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
        const scrollY = window.pageYOffset || document.documentElement.scrollTop;
        scrollProgress = Math.min(1, Math.max(0, scrollY / maxScroll));

        // Map scrollProgress (0.0 to 1.0) into the 5 keyframe segments
        const numSegments = sceneKeyframes.length - 1;
        const scaled = scrollProgress * numSegments;
        const index = Math.min(Math.floor(scaled), numSegments - 1);
        const t = scaled - index; // local segment progress 0.0 to 1.0

        // Smooth cubic easing for seamless transitions between chapters
        const easeT = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

        const kfA = sceneKeyframes[index];
        const kfB = sceneKeyframes[index + 1];

        // Interpolate camera position and lookAt target
        targetCamPos.lerpVectors(kfA.camPos, kfB.camPos, easeT);
        targetLookAt.lerpVectors(kfA.lookAt, kfB.lookAt, easeT);

        // Interpolate probe position
        if (guidingProbeGroup) {
            guidingProbeGroup.position.lerpVectors(kfA.probePos, kfB.probePos, easeT);
        }

        // Interpolate specimen position & scale
        if (specimenGroup) {
            specimenGroup.position.lerpVectors(kfA.specimenPos, kfB.specimenPos, easeT);
            const scale = THREE.MathUtils.lerp(kfA.specimenScale, kfB.specimenScale, easeT);
            specimenGroup.scale.setScalar(scale);
            // In Scene 04, specimen is tucked away so Intelligence Network takes center stage
            specimenGroup.visible = (index < 3 || (index === 3 && t < 0.25));
        }

        // Interpolate Machine Learning Intelligence Network
        if (intelligenceGroup) {
            intelligenceGroup.position.lerpVectors(kfA.intelPos, kfB.intelPos, easeT);
            const intScale = THREE.MathUtils.lerp(kfA.intelScale, kfB.intelScale, easeT);
            intelligenceGroup.scale.setScalar(intScale);
            const intOp = THREE.MathUtils.lerp(kfA.intelOpacity, kfB.intelOpacity, easeT);
            intelligenceGroup.visible = (intOp > 0.05);
        }

        // Update active chapter HUD indicator
        updateChapterHUD(index, t);
    }

    let lastAudioChapter = -1;
    function updateChapterHUD(chapterIndex, localProgress) {
        const chapter = (localProgress > 0.5) ? chapterIndex + 1 : chapterIndex;
        if (chapter !== lastAudioChapter) {
            if (lastAudioChapter !== -1 && window.AMRAudio?.chapterTransition) {
                window.AMRAudio.chapterTransition(chapter);
            }
            lastAudioChapter = chapter;
        }
        document.querySelectorAll('.chapter-nav-item').forEach((item, idx) => {
            const isActive = idx === chapter;
            item.classList.toggle('active', isActive);
            if (isActive) item.setAttribute('aria-current', 'step');
            else item.removeAttribute('aria-current');
        });

        const progressFill = document.getElementById('chapterProgressFill');
        if (progressFill) {
            progressFill.style.width = `${(scrollProgress * 100).toFixed(1)}%`;
        }
    }

    // ─── Mouse Movement & Specimen Drag Controls ────────────────────────────────
    function onMouseMove(e) {
        const halfX = window.innerWidth / 2;
        const halfY = window.innerHeight / 2;
        mouse.targetX = (e.clientX - halfX) * 0.04;
        mouse.targetY = (e.clientY - halfY) * 0.04;

        if (isDraggingSpecimen) {
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            targetSpecimenRotation.y += deltaX * 0.008;
            targetSpecimenRotation.x += deltaY * 0.008;

            previousMousePosition = { x: e.clientX, y: e.clientY };
        }
    }

    function setupSpecimenDragControls() {
        const specimenStage = document.getElementById('specimenInteractiveStage');
        if (!specimenStage) return;

        specimenStage.addEventListener('mousedown', (e) => {
            isDraggingSpecimen = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
            specimenStage.style.cursor = 'grabbing';
        });

        specimenStage.addEventListener('dblclick', () => {
            targetSpecimenRotation.x = 0.2;
            targetSpecimenRotation.y = 0.4;
        });

        window.addEventListener('mouseup', () => {
            if (isDraggingSpecimen) {
                isDraggingSpecimen = false;
                if (specimenStage) specimenStage.style.cursor = 'grab';
            }
        });

        // Touch support for tablet / mobile
        specimenStage.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                isDraggingSpecimen = true;
                previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
            }
        }, { passive: true });

        window.addEventListener('touchmove', (e) => {
            if (isDraggingSpecimen && e.touches.length === 1) {
                const deltaX = e.touches[0].clientX - previousMousePosition.x;
                const deltaY = e.touches[0].clientY - previousMousePosition.y;

                targetSpecimenRotation.y += deltaX * 0.01;
                targetSpecimenRotation.x += deltaY * 0.01;

                previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
            }
        }, { passive: true });

        window.addEventListener('touchend', () => {
            isDraggingSpecimen = false;
        });
    }

    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function onVisibilityChange() {
        if (document.hidden) {
            stopLoop();
        } else if (isLandingActive && !prefersReducedMotion) {
            startLoop();
        }
    }

    // ─── Main Animation Loop ───────────────────────────────────────────────────
    function animate() {
        if (!isLandingActive) return;

        animFrameId = requestAnimationFrame(animate);

        const delta = clock.getDelta();
        const time = clock.getElapsedTime();

        // 1. Smooth Camera Interpolation (Apple-style smooth lerp)
        mouse.x += (mouse.targetX - mouse.x) * 0.05;
        mouse.y += (mouse.targetY - mouse.y) * 0.05;

        currentCamPos.lerp(targetCamPos, 0.06);
        currentLookAt.lerp(targetLookAt, 0.06);

        camera.position.x = currentCamPos.x + mouse.x * 0.4;
        camera.position.y = currentCamPos.y - mouse.y * 0.4;
        camera.position.z = currentCamPos.z;
        camera.lookAt(currentLookAt);

        // 2. Pulse Guiding Light Core
        if (guidingProbeGroup) {
            const pulse = 1 + Math.sin(time * 2.2) * 0.12;
            guidingProbeGroup.scale.set(pulse, pulse, pulse);
            if (guidingLight) {
                guidingLight.intensity = 2.8 + Math.sin(time * 2.5) * 0.6;
            }
        }

        // 3. Specimen Rotation & Inertia Damping (Scene 03)
        if (specimenGroup && specimenGroup.visible) {
            if (!isDraggingSpecimen) {
                targetSpecimenRotation.y += 0.0035;
            }

            specimenRotation.x += (targetSpecimenRotation.x - specimenRotation.x) * 0.08;
            specimenRotation.y += (targetSpecimenRotation.y - specimenRotation.y) * 0.08;

            specimenGroup.rotation.x = specimenRotation.x;
            specimenGroup.rotation.y = specimenRotation.y;

            if (coreHelixGroup) {
                coreHelixGroup.rotation.z = time * 0.45;
                coreHelixGroup.rotation.y = time * 0.2;
            }
            if (receptorMeshGroup) {
                receptorMeshGroup.rotation.z = -time * 0.25;
            }
            if (outerLatticeMesh) {
                const memPulse = 1 + Math.sin(time * 1.6) * 0.012;
                outerLatticeMesh.scale.set(memPulse, memPulse, memPulse);
            }
        }

        // 3B. Machine Learning Intelligence Network Rotation & Synaptic Pulses (Scene 04)
        if (intelligenceGroup && intelligenceGroup.visible) {
            intelligenceGroup.rotation.y = time * 0.14;
            if (intelligenceCore) {
                intelligenceCore.rotation.x = time * 0.22;
                intelligenceCore.rotation.z = time * 0.16;
                const cPulse = 1 + Math.sin(time * 2.8) * 0.05;
                intelligenceCore.scale.set(cPulse, cPulse, cPulse);
            }
            if (targetNodesGroup) {
                targetNodesGroup.children.forEach((node, i) => {
                    const nPulse = 1 + Math.sin(time * 2.5 + i * 0.78) * 0.1;
                    node.scale.set(nPulse, nPulse, nPulse);
                });
            }
        }

        // 4. Bio-fluid Drift for Cellular Markers
        if (bioParticlePoints) {
            bioParticlePoints.rotation.y = time * 0.012;
            bioParticlePoints.rotation.z = time * 0.006;
        }

        renderer.render(scene, camera);
    }

    function startLoop() {
        if (!animFrameId && isLandingActive) {
            clock.start();
            animate();
        }
    }

    function stopLoop() {
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
    }

    // ─── Public API for Spectroscopy & Layer Toggles ───────────────────────────
    window.AMR3D = {
        // Switch between the 5 Landing Scenes
        jumpToScene: function (sceneIndex) {
            if (sceneIndex < 0 || sceneIndex >= sceneKeyframes.length) return;

            const targetSection = document.getElementById(`landing-scene-0${sceneIndex + 1}`);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
                if (window.AMRAudio && window.AMRAudio.chime) {
                    window.AMRAudio.chime(440 + sceneIndex * 80, 0.8);
                }
            }
        },

        // Toggle Specimen Structural Layers
        toggleLayer: function (layerId, isVisible) {
            if (layerId === 'porins' && porinMeshGroup) {
                porinMeshGroup.visible = isVisible;
            } else if (layerId === 'receptors' && receptorMeshGroup) {
                receptorMeshGroup.visible = isVisible;
            } else if ((layerId === 'matrix' || layerId === 'membrane') && outerLatticeMesh) {
                outerLatticeMesh.visible = isVisible;
            } else if (layerId === 'core' && coreHelixGroup) {
                coreHelixGroup.visible = isVisible;
            } else if (layerId === 'peptidoglycan' && peptidoglycanGroup) {
                peptidoglycanGroup.visible = isVisible;
            }
            if (window.AMRAudio && window.AMRAudio.chime) {
                window.AMRAudio.chime(560, 0.5);
            }
        },

        // Switch Spectroscopy Illumination Modes
        setSpectroscopy: function (mode) {
            if (!materials.confocalLattice) return;

            if (mode === 'confocal') {
                // Neon Cyan / Laboratory Teal
                materials.confocalLattice.color.setHex(0x00F2C3);
                materials.confocalLattice.emissive.setHex(0x0369A1);
                if (materials.membraneLipid) materials.membraneLipid.color.setHex(0x0284C7);
                materials.confocalWire.color.setHex(0x00D2FF);
                if (materials.porin) materials.porin.color.setHex(0x38BDF8);
                if (materials.receptor) materials.receptor.color.setHex(0xF59E0B);
                if (materials.coreHelix) materials.coreHelix.color.setHex(0x818CF8);
                scene.fog.color.setHex(0x040711);
            } else if (mode === 'darkfield') {
                // Deep Electric Sapphire & High-Scatter Edge
                materials.confocalLattice.color.setHex(0x38BDF8);
                materials.confocalLattice.emissive.setHex(0x0284C7);
                if (materials.membraneLipid) materials.membraneLipid.color.setHex(0x081b3b);
                materials.confocalWire.color.setHex(0x818CF8);
                if (materials.porin) materials.porin.color.setHex(0x00F2C3);
                if (materials.receptor) materials.receptor.color.setHex(0x38BDF8);
                if (materials.coreHelix) materials.coreHelix.color.setHex(0x6366F1);
                scene.fog.color.setHex(0x010308);
            } else if (mode === 'electron') {
                // Cryo-EM Monochromatic Silver / Platinum Contrast
                materials.confocalLattice.color.setHex(0xE2E8F0);
                materials.confocalLattice.emissive.setHex(0x334155);
                if (materials.membraneLipid) materials.membraneLipid.color.setHex(0x1e293b);
                materials.confocalWire.color.setHex(0x94A3B8);
                if (materials.porin) materials.porin.color.setHex(0xF1F5F9);
                if (materials.receptor) materials.receptor.color.setHex(0xE2E8F0);
                if (materials.coreHelix) materials.coreHelix.color.setHex(0xCBD5E1);
                scene.fog.color.setHex(0x05070D);
            }

            if (window.AMRAudio && window.AMRAudio.chime) {
                window.AMRAudio.chime(680, 0.6);
            }
        },

        // Switch page views (Landing vs Internal App Shell)
        setPageScene: function (pageId) {
            if (pageId === 'page-landing') {
                isLandingActive = true;
                canvas.style.display = 'block';
                canvas.style.opacity = '1';
                startLoop();
                onScroll();
            } else {
                // When in clinical workspace, hide 3D canvas and halt loop to prevent chart obstruction
                isLandingActive = false;
                canvas.style.display = 'none';
                canvas.style.opacity = '0';
                stopLoop();
            }
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
