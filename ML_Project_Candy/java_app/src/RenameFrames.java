import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Programme pour renommer les images extraites avec des noms uniques
 */
public class RenameFrames {
    
    public static void main(String[] args) {
        String videoFolderPath = "/media/pc_remi/Synchro/ESISAR/3_Semestre/IN451/Bonbons/video/Photos-3-001";
        
        File videoFolder = new File(videoFolderPath);
        
        if (!videoFolder.exists() || !videoFolder.isDirectory()) {
            System.err.println("❌ Le dossier spécifié n'existe pas : " + videoFolderPath);
            return;
        }
        
        // Trouver tous les dossiers contenant "_frames"
        File[] frameFolders = videoFolder.listFiles((dir, name) -> 
            name.contains("_frames") && new File(dir, name).isDirectory()
        );
        
        if (frameFolders == null || frameFolders.length == 0) {
            System.out.println("⚠️  Aucun dossier de frames trouvé.");
            return;
        }
        
        System.out.println("📂 " + frameFolders.length + " dossier(s) de frames trouvé(s)");
        System.out.println("🔄 Renommage en cours...\n");
        
        int totalRenamed = 0;
        
        for (File frameFolder : frameFolders) {
            int renamed = renameFramesInFolder(frameFolder);
            totalRenamed += renamed;
        }
        
        System.out.println("\n✅ Renommage terminé !");
        System.out.println("📊 Total : " + totalRenamed + " images renommées");
    }
    
    private static int renameFramesInFolder(File frameFolder) {
        String folderName = frameFolder.getName();
        
        // Extraire le nom de base (tout ce qui est avant _frames)
        String baseName = folderName.split("_frames")[0];
        
        System.out.println("📁 Traitement de : " + folderName);
        
        // Trouver tous les fichiers .jpg
        File[] imageFiles = frameFolder.listFiles((dir, name) -> 
            name.toLowerCase().endsWith(".jpg")
        );
        
        if (imageFiles == null || imageFiles.length == 0) {
            System.out.println("   ⚠️  Aucune image trouvée");
            return 0;
        }
        
        // Trier les fichiers par nom pour un ordre cohérent
        java.util.Arrays.sort(imageFiles, (a, b) -> a.getName().compareTo(b.getName()));
        
        int count = 0;
        for (File imageFile : imageFiles) {
            // Nouveau nom : baseName_frame_0000.jpg
            String newName = String.format("%s_frame_%04d.jpg", baseName, count);
            File newFile = new File(frameFolder, newName);
            
            try {
                // Renommer le fichier
                Files.move(imageFile.toPath(), newFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                count++;
            } catch (IOException e) {
                System.err.println("   ❌ Erreur lors du renommage : " + imageFile.getName());
                e.printStackTrace();
            }
        }
        
        System.out.println("   ✅ " + count + " images renommées");
        return count;
    }
}
