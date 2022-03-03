classdef Directories

    properties (Constant)
        root = pwd;
        ROSBagDir = [pwd '/ROSBag'];
        rawDataDir = [pwd '/rawData'];
        dataDir = [pwd '/data'];
        demonstrationDir = [Directories.dataDir '/demonstrations'];
        videoDir = [Directories.dataDir '/videos'];
        classifierRenderDir = [Directories.dataDir '/classifierRender'];
        blenderRenderDir = [Directories.dataDir '/render'];
        blenderDir = [pwd '/blender'];
        URDFDir = [pwd '/URDF'];
        cachedDir = [pwd '/cached'];
        trainedNetworkDir = [Directories.cachedDir '/trainedNetworks'];
        differentialRendererDataDir = [Directories.dataDir '/differentialRenderer'];
        differentialRendererDir = [pwd '/differentialRenderer']
        sceneDir = [pwd '/scenes'];
        objsDir = [Directories.sceneDir '/objects'];
        SegmentationClassifier = [pwd '/renderImageBasedSegmentation'];
    end

end