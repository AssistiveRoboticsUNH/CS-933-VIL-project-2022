function makeVideos
Utils.removeDir(Directories.videoDir)

Utils.makeDirIfMissing(Directories.videoDir);
demonstrations = Utils.getAllFileNames(Directories.demonstrationDir);
for d = 1:length(demonstrations)
    demo = demonstrations{d};
    files = Utils.getAllFileNames(demo, 'filter', @(x) (contains(x,'joint') || contains(x,'Mask')));
    outputVideo = VideoWriter([Directories.videoDir '/' num2str(d) '.avi'],'Uncompressed AVI');
    outputVideo.FrameRate = 30;
    open(outputVideo)
    for ii = 1:length(files)
        file = files{ii};
        img = imread(file);
        writeVideo(outputVideo,img)
    end
    close(outputVideo)
end

end