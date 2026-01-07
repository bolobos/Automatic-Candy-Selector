import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Programme pour extraire 5 images par seconde de vidéos
 */
public class VideoFrameExtractor {
    
    static {
        System.loadLibrary("opencv_java4100");
    }
    
    private static final int FRAMES_PER_SECOND = 5;
    
    public static void main(String[] args) {
        String videoFolderPath = "/media/pc_remi/Synchro/ESISAR/3_Semestre/IN451/Bonbons/video/Photos-3-001";
        
        File videoFolder = new File(videoFolderPath);
        
        if (!videoFolder.exists() || !videoFolder.isDirectory()) {
            System.err.println("❌ Le dossier spécifié n'existe pas : " + videoFolderPath);
            return;
        }
        
        File[] videoFiles = videoFolder.listFiles((dir, name) -> {
            String lowerName = name.toLowerCase();
            return lowerName.endsWith(".mp4") || lowerName.endsWith(".avi") || 
                   lowerName.endsWith(".mov") || lowerName.endsWith(".mkv") ||
                   lowerName.endsWith(".wmv") || lowerName.endsWith(".flv");
        });
        
        if (videoFiles == null || videoFiles.length == 0) {
            System.out.println("⚠️  Aucune vidéo trouvée dans le dossier.");
            return;
        }
        
        System.out.println("📹 " + videoFiles.length + " vidéo(s) trouvée(s)");
        System.out.println("🎯 Extraction de " + FRAMES_PER_SECOND + " images par seconde\n");
        
        for (File videoFile : videoFiles) {
            extractFrames(videoFile, videoFolderPath);
        }
        
        System.out.println("\n✅ Extraction terminée !");
    }
    
    private static void extractFrames(File videoFile, String basePath) {
        String videoName = videoFile.getName();
        String videoNameWithoutExt = videoName.substring(0, videoName.lastIndexOf('.'));
        
        // Créer le dossier de sortie pour cette vidéo
        String outputFolder = basePath + File.separator + videoNameWithoutExt + "_frames";
        try {
            Files.createDirectories(Paths.get(outputFolder));
        } catch (Exception e) {
            System.err.println("❌ Erreur lors de la création du dossier : " + outputFolder);
            e.printStackTrace();
            return;
        }
        
        System.out.println("📹 Traitement de : " + videoName);
        
        VideoCapture capture = new VideoCapture(videoFile.getAbsolutePath());
        
        if (!capture.isOpened()) {
            System.err.println("❌ Impossible d'ouvrir la vidéo : " + videoName);
            return;
        }
        
        // Obtenir le FPS de la vidéo
        double fps = capture.get(Videoio.CAP_PROP_FPS);
        int totalFrames = (int) capture.get(Videoio.CAP_PROP_FRAME_COUNT);
        double duration = totalFrames / fps;
        
        System.out.println("   FPS: " + fps + " | Durée: " + String.format("%.2f", duration) + "s | Total frames: " + totalFrames);
        
        // Calculer l'intervalle entre les frames à extraire
        int frameInterval = (int) Math.round(fps / FRAMES_PER_SECOND);
        if (frameInterval < 1) frameInterval = 1;
        
        Mat frame = new Mat();
        int frameCount = 0;
        int extractedCount = 0;
        
        while (capture.read(frame)) {
            if (frameCount % frameInterval == 0) {
                String outputPath = outputFolder + File.separator + 
                                  String.format("frame_%04d.jpg", extractedCount);
                
                if (Imgcodecs.imwrite(outputPath, frame)) {
                    extractedCount++;
                } else {
                    System.err.println("   ⚠️  Erreur d'écriture : " + outputPath);
                }
            }
            frameCount++;
            
            // Afficher la progression tous les 10%
            if (frameCount % (totalFrames / 10) == 0) {
                int progress = (int) ((frameCount * 100.0) / totalFrames);
                System.out.println("   📊 Progression : " + progress + "%");
            }
        }
        
        capture.release();
        frame.release();
        
        System.out.println("   ✅ " + extractedCount + " images extraites dans : " + outputFolder + "\n");
    }
}
