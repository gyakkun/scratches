import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.NoSuchAlgorithmException;

// DES 实验
class Scratch {

    private static Cipher cipherEncrypt;
    private static Cipher cipherDecrypt;

    public static void main(String[] args) throws NoSuchPaddingException, NoSuchAlgorithmException, InvalidKeyException, BadPaddingException, IllegalBlockSizeException {

        String key = "00000000";
        Key keyForCipher = new SecretKeySpec(key.getBytes(), "DES");

        // 加密的cipher
        cipherEncrypt = Cipher.getInstance("DES/ECB/NoPadding");
        cipherEncrypt.init(Cipher.ENCRYPT_MODE, keyForCipher);
        byte[] byteToEncrypt = hexStrToBytes("017E210207062112");
        byte[] encryptedMsg = cipherEncrypt.doFinal(byteToEncrypt);
//        System.err.println(new String(encryptedMsg));
        System.err.println(bytesToHexStr(encryptedMsg));

        // 解密的cipher
        cipherDecrypt = Cipher.getInstance("DES/ECB/NoPadding");
        cipherDecrypt.init(Cipher.DECRYPT_MODE, keyForCipher);
        byte[] decryptedMsg = cipherDecrypt.doFinal(encryptedMsg);
        System.err.println(bytesToHexStr(decryptedMsg));


    }

    private static String bytesToHexStr(byte[] bs) {
        String result = "";
        char[] hex = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
        for (byte b : bs) {
            int a = b & 0xff;
            char high = hex[a >> 4];
            char low = hex[((a << 4) & 0xff) >> 4];
            result += high;
            result += low;
            result += " ";
        }
        return result;
    }

    private static byte[] hexStrToBytes(String hexStr) {
        byte[] result = new byte[hexStr.length() >> 1];
        for (int i = 0; i < result.length; i++) {
            String tmp = hexStr.substring(2 * i, 2 * i + 2);
            int tmpInt = Integer.parseInt(tmp, 16);
            result[i] = (byte) tmpInt;
        }
        return result;
    }

}