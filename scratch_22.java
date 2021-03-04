// MyDes
import javax.crypto.Cipher;

class Scratch {
    //初始置换表IP
    private byte IP_Table[] = {
            58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7
    };
    // IP-1置换表
    private byte IPR_Table[] = {
            40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25
    };
    // E扩展表
    private byte E_Table[] = {
            32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1
    };
    // PC1置换表
    private byte[] PC1_Table = {
            57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4
    };
    // pc2表
    private byte PC2_Table[] = {
            14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55, 30, 40, 51, 34, 33, 48,
            44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32
    };
    //  移位表
    private byte Move_Table[] = {
            1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
    };
    // S盒
    private byte Sbox1[][] = {{14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7},
            {0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
            {4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,},
            {15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13,}};
    private byte Sbox2[][] = {{15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,},
            {3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,},
            {0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,},
            {13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9,}};
    private byte Sbox3[][] = {{10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,},
            {13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,},
            {13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7},
            {1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12,}};
    private byte Sbox4[][] = {{7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,},
            {13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,},
            {10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,},
            {3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14,}};
    private byte Sbox5[][] = {{2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,},
            {14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,},
            {4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,},
            {11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3,}};
    private byte Sbox6[][] = {{12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,},
            {10, 15, 4, 2, 7, 12, 0, 5, 6, 1, 13, 14, 0, 11, 3, 8,},
            {9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,},
            {4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13,}};
    private byte Sbox7[][] = {{4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,},
            {13, 0, 11, 7, 4, 0, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,},
            {1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,},
            {6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12,}};
    private byte Sbox8[][] = {{13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,},
            {1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,},
            {7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,},
            {2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}};
    //P置换表
    private byte P_Table[] = {
            16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
            2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25
    };
    private static byte SubKey[][] = new byte[16][48];

    byte[] byteToBitReverse(byte[] out, byte[] ori, int num) {
        for (int i = 0; i < num; i++) {
            out[i % 4 + 3] = (byte) ((ori[i / 4] >> (i % 4)) & 0x01);
        }
        return out;
    }

    byte[] byteToBit(byte[] out, byte[] ori, int num) {
        for (int i = 0; i < num; i++) {
            out[i] = (byte) ((ori[i / 8] >> (i % 8)) & 0x01);
        }
        return out;
    }

    byte[] byteToBit(byte[] ori, int num) {
        byte[] res = new byte[num];
        for (int i = 0; i < num; i++) {
            res[i] = (byte) ((ori[i / 8] >> (i % 8)) & 0x01);
        }
        return res;
    }

    byte[] byteToBitOri(byte[] out, byte[] ori, int num) {
        for (int i = 0; i < num; i++) {
            out[i] = (byte) ((ori[i / 8] >> (i % 8)) & 0x01);
        }
        return out;
    }

    //二进制数组拷贝
    byte[] bitCopy(byte[] ori, int num) {
        byte[] dest = new byte[num];
        System.arraycopy(ori, 0, dest, 0, num);
        return dest;
    }

    byte[] bitCopyFrom(byte[] ori, int num, int from) {
        byte[] dest = new byte[num];
        System.arraycopy(ori, from, dest, 0, num);
        return dest;
    }

    byte[] bitCopyNew(byte[] out, byte[] ori, int num) {
        System.arraycopy(ori, 0, out, 0, num);
        return out;
    }

    byte[] bitToByte(byte[] in, int num) {
        byte[] out = new byte[num / 8];
        int i = 0;
        for (i = 0; i < num; i++) {
            out[i / 8] = (byte) (out[i / 8] | in[i] << (i % 8));
        }
        return out;
    }

    int[] bitToInt(byte[] in, int num) {
        int[] out = new int[num / 32];
        int i = 0;
        for (i = 0; i < num; i++) {
            out[i / 32] = (byte) (out[i / 32] | in[i] << (i % 8));
        }
        return out;
    }

    /* public static byte[] ByteToBit (byte a) {
         byte[] temp= new byte[8];
         for (int i = 7; i >= 0; i--) {
             temp[i] = (byte)((a >> i) & 1);
         }
         return temp;
     }*/
    public byte[] BitToByte(byte[] a, int num) {
        byte[] res = new byte[a.length / num];
        byte temp = (byte) 0;
        for (int j = 0; j < a.length / num; j++) {
            for (int i = 0; i < a.length; i++) {
                temp = (byte) (temp | a[i] << i);
                if (i % 8 == 0) {
                    res[j] = temp;
                    temp = 0;
                }
            }
        }
        return res;
    }

    byte[] tableReplace(byte[] ori, byte[] table, int num) {
//        byte[] res = new byte[];
        byte[] temp = new byte[num];
        for (int i = 0; i < num; i++) {
//            System.out.println("tablei:"+table[i]+", oritable:"+ori[table[i]-1]);
            temp[i] = ori[table[i] - 1];
        }
//        for(int i=num; i< ori.length; i++){
//            temp[i] = ori[i];
//        }
        return temp;
    }

    /*byte[] loopBit(byte[] ori, int moveStep, int len){
        byte[] temp = new byte[256];
        temp =bitCopy(ori, moveStep);
        byte[] next = new byte[ori.length-moveStep];
        System.arraycopy(ori, moveStep,next, 0, ori.length - moveStep);
        byte[] next2 = bitCopy(next, len - moveStep);
//        bitCopy()
        return null;
    }*/

    //已测
    byte[] loopBit2(byte[] ori, int moveStep, int len) {
        byte[] res = new byte[ori.length];
        for (int i = 0, j = moveStep; i < len; i++) {
            if (i < len - moveStep) {
                res[i] = ori[j++];
            } else {
                res[i] = ori[j++ - len];
            }
        }
        return res;
    }

    byte[] bitToHex(byte[] Data_out, byte[] Data_in, int Num) //二进制转十六进制
    {
        int i;
        for (i = 0; i < Num / 4; i++) {
            Data_out[i] = 0;
        }
        for (i = 0; i < Num / 4; i++) {
            Data_out[i] = (byte) (Data_in[4 * i] + Data_in[4 * i + 1] * 2 + Data_in[4 * i + 2] * 4 + Data_in[4 * i + 3] * 8);
            if (Data_out[i] % 16 > 9) {
                Data_out[i] = (byte) (Data_out[i] % 16 + '7');
            } else
                Data_out[i] = (byte) (Data_out[i] % 16 + '0');
        }
        return Data_out;
    }

    //执行异或
    byte[] xor(byte[] out, byte[] bits, int num) {
        byte[] res = new byte[out.length];
        for (int i = 0; i < num; i++) {
            res[i] = (byte) (out[i] ^ bits[i]);
        }
        return res;
    }

    byte[] xor2(byte[] out, byte[] bits, int num) {
        //byte[] res = new byte[num];
        for (int i = 0; i < num; i++) {
            out[i] = (byte) (out[i] ^ bits[i]);
        }
        return out;
    }

    void setKey(byte[] myKey) {
        byte[] keyBit = new byte[64];
        keyBit = byteToBit(keyBit, myKey, 64);
        keyBit = tableReplace(keyBit, PC1_Table, 56);
//        for(int i=0; i<keyBit.length; i++){
//            System.out.print(keyBit[i]);
//        }
//        System.out.println(keyBit.length);
        byte[] keyBitL = new byte[28], keyBitR = new byte[28], keyBitE = new byte[8];
        System.arraycopy(keyBit, 0, keyBitL, 0, 28);
        System.arraycopy(keyBit, 28, keyBitR, 0, 28);
//        System.arraycopy(keyBit,56,keyBitE,0,8);
        for (int i = 0; i < 16; i++) {
            keyBitL = loopBit2(keyBitL, Move_Table[i], 28);
            keyBitR = loopBit2(keyBitR, Move_Table[i], 28);
//            keyBit = concat(concat(keyBitL, keyBitR),keyBitE);
            keyBit = concat(keyBitL, keyBitR);
            SubKey[i] = tableReplace(keyBit, PC2_Table, 48);
            for (int k = 0; k < SubKey[i].length; k++) {
//                System.out.print(SubKey[i][k]);
            }
        }
    }


    /**
     * 正确的10进制转二进制
     *
     * @param in
     * @param num
     * @return
     */
    byte[] hexTOBit3(byte[] in, int num) {
        byte[] result = new byte[num];
        int k = 0;
        for (int j = 0; j < in.length; j++) {
            for (int i = 7; i >= 0; i--) {
                result[k++] = (byte) (in[j] >>> i & 1);
            }
        }
        return result;
    }

    /**
     * bad转二进制,每四个位相反
     *
     * @param in
     * @param num
     * @return
     */
    byte[] hexTOBit4(byte[] in, int num) {
        byte[] result = new byte[num];
        int k = 0;
        for (int j = 0; j < in.length; j++) {
            for (int i = 7; i >= 0; i--, k++) {
                if (k % 4 == 0) {
                    result[k + 3] = (byte) (in[j] >>> i & 1);
                }
                if (k % 4 == 1) {
                    result[k + 1] = (byte) (in[j] >>> i & 1);
                }
                if (k % 4 == 2) {
                    result[k - 1] = (byte) (in[j] >>> i & 1);
                }

                if (k % 4 == 3) {
                    result[k - 3] = (byte) (in[j] >>> i & 1);
                }
            }
        }
        return result;
    }

    byte[] hexToBit(byte[] in, int num) {
        byte[] out = new byte[num];
        for (int i = 0; i < num; i++) {
            if (in[i / 8] <= 9) {
                out[i] = (byte) (((in[i / 8] - 0) >> (i % 8)) & 0x01);
            } else {
                out[i] = (byte) (((in[i / 8] - 7) >> (i % 8)) & 0x01);
            }
        }
        return out;
    }

    byte[] runDesDe(byte[] plain, byte[] chiperText) {
        byte[] msgBit = new byte[64];
        byte[] msgBitL = new byte[32];
        byte[] msgBitR = new byte[32];
        byte[] temp = new byte[32];
        msgBit = hexTOBit4(chiperText, 64);
//        msgBit = byteToBitReverse(msgBit,chiperText, 64);

//        sou(msgBit);
//        msgBit = new byte[]{0,1,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,1,1,1};
        msgBit = tableReplace(msgBit, IP_Table, 64);
        msgBitR = bitCopyFrom(msgBit, 32, 32);
        msgBitL = bitCopyFrom(msgBit, 32, 0);
        for (int i = 15; i >= 0; i--) {
            temp = bitCopyNew(temp, msgBitL, 32);
            msgBitL = fChange(msgBitL, SubKey[i]);
            msgBitL = xor(msgBitL, msgBitR, 32);
            msgBitR = bitCopyNew(msgBitR, temp, 32);
            msgBit = concat(msgBitL, msgBitR);
        }
        msgBit = tableReplace(msgBit, IPR_Table, 64);
        plain = bitToByte(msgBit, 64);
        return plain;
    }

    static byte[] concat(byte[] a, byte[] b) {
        byte[] c = new byte[a.length + b.length];
        System.arraycopy(a, 0, c, 0, a.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    byte[] sChange(byte[] out, byte[] in) {
//        byte[] out = new byte[32];
        int i = 0, r = 0, c = 0;
        int j = 0; //in输入的下标 s盒
        for (i = 0; i < 8; i++, j += 6) {
            r = in[j] * 2 + in[j + 5] * 1;
            c = in[j + 1] * 8 + in[j + 2] * 4 + in[j + 3] * 2 + in[j + 4] * 1;

            switch (i) {
                case 0:
                    byte[] bytes = byteToBit(new byte[]{Sbox1[r][c]}, 4);
                    System.arraycopy(bytes, 0, out, i * 4, 4);
//                    byteToBit(out,bytes,4);
                    break;
                case 1:
                    byte[] bytes1 = byteToBit(new byte[]{Sbox2[r][c]}, 4);
                    System.arraycopy(bytes1, 0, out, i * 4, 4);
//                    byteToBit(out,bytes1,4);
                    break;
                case 2:
                    byte[] bytes2 = byteToBit(new byte[]{Sbox3[r][c]}, 4);
                    System.arraycopy(bytes2, 0, out, i * 4, 4);
//                    byteToBit(out,bytes2,4);
                    break;
                case 3:
                    byte[] byte3 = byteToBit(new byte[]{Sbox4[r][c]}, 4);
                    System.arraycopy(byte3, 0, out, i * 4, 4);
//                    byteToBit(out,byte3,4);
                    break;
                case 4:
                    byte[] bytes4 = byteToBit(new byte[]{Sbox5[r][c]}, 4);
                    System.arraycopy(bytes4, 0, out, i * 4, 4);
//                    byteToBit(out,bytes4,4);
                    break;
                case 5:
                    byte[] bytes5 = byteToBit(new byte[]{Sbox6[r][c]}, 4);
                    System.arraycopy(bytes5, 0, out, i * 4, 4);
//                    byteToBit(out,bytes5,4);
                    break;
                case 6:
                    byte[] bytes6 = byteToBit(new byte[]{Sbox7[r][c]}, 4);
                    System.arraycopy(bytes6, 0, out, i * 4, 4);
//                    byteToBit(out,bytes6,4);
                    break;
                case 7:
                    byte[] bytes7 = byteToBit(new byte[]{Sbox8[r][c]}, 4);
                    System.arraycopy(bytes7, 0, out, i * 4, 4);
//                    byteToBit(out,bytes7,4);
                    break;
            }
        }
        return out;
    }

    byte[] fChange(byte[] dataout, byte[] in) {
        byte[] msge = new byte[48];
        msge = tableReplace(dataout, E_Table, 48);
//        if(i>12){        sou(msge);sou(dataout);}
        byte[] arrxor = xor(msge, in, 48);
//        if(i>12){        sou(arrxor);}
        byte[] out = sChange(dataout, arrxor);
        //if(i>12){sou(arrxor);sou(out); }
        byte[] bytes = tableReplace(out, P_Table, 32);
        return bytes;
    }


    /**
     * @param plaintext 8
     * @param key       8
     * @return
     */
    byte[] des_en(byte[] plaintext, byte[] key) {
        byte[] hex = new byte[16];
        setKey(key);
//        System.out.println("ori plantext:"+StringUtil.toHexStringPadded(plaintext));
        byte[] bytes = runDes(plaintext, hex);
        System.out.println("DES_Encryption:" + new String(bytes));
//        System.out.println("hex:"+StringUtil.toHexStringPadded(bytes));
        return bytes;
    }

    /**
     * @param plaintext  8
     * @param ciphertext 8
     * @param key        8
     * @return
     */
    byte[] des_de(byte[] plaintext, byte[] ciphertext, byte[] key) {
        setKey(key);
        byte[] temp = new byte[16];
        byte[] hex = new byte[16];
      /*  for(int i=0; i<8; i++){
            hex[i*2] = (byte) ((ci[i]>>8)&0xff);
            hex[i*2+1] = (byte) (ci[i]&0xff);
        }*/
//        hex = new byte[]{(byte) 0xea,0x55,0x45,0x5e, (byte) 0xb4,0x31,0x50,0x5f};
//        hex = new byte[]{5,5,4,5,5,1,2,3,4,5,6,7,8,9,8,7};
        byte[] bytes = runDesDe(plaintext, ciphertext);
        System.out.println("DES_Decryption" + new String(bytes));
//        System.out.println("hex:"+StringUtil.toHexStringPadded(bytes));
        return bytes;
    }


    /**
     * @param myMsg 8
     * @param out   16
     */
    byte[] runDes(byte[] myMsg, byte[] out) {
        byte[] messageBit = new byte[64];
        byte[] temp = new byte[32];
        byte[] msgbitR = new byte[32];
        byte[] msgbitL = new byte[32];
        messageBit = byteToBit(messageBit, myMsg, 64);
        messageBit = tableReplace(messageBit, IP_Table, 64);
        msgbitR = bitCopyFrom(messageBit, 32, 32);
        msgbitL = bitCopyFrom(messageBit, 32, 0);
        System.out.println("hex bit:");
        for (int i = 0; i < 16; i++) {
            temp = bitCopyNew(temp, msgbitR, 32);
//            if(i>12){sou(msgbitR);}
            msgbitR = fChange(msgbitR, SubKey[i]);

//            if(i>12){sou(msgbitR);}
            msgbitR = xor(msgbitR, msgbitL, 32);
            bitCopyNew(msgbitL, temp, 32);
            messageBit = concat(msgbitL, msgbitR);

        }
        messageBit = tableReplace(messageBit, IPR_Table, 64);
        out = bitToHex(out, messageBit, 64);
        return out;
    }

//    public static void main(String[] args) {
//        Scratch des = new Scratch();
//        String a = "13100098FF150813";
//        try {
////            byte[] text = StringUtil.decodeHexDump(a);
//            byte mykey[] = {0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30};
//
//            String plain = "A3C1FF5338827B08";
//            byte[] hexMsg = new byte[16];
////            byte[] runDes = des.runDes(mykey, text);
////            byte[] bytes = des.des_en(text, mykey, mykey);
//            byte[] bytes = des.des_en(mykey, "25217370".getBytes());
//            System.err.println(new String(bytes));
//
////            A6 40 2B 0F 2D 88 55 0F
//            byte mykey2[] = {(byte) 0xa6, (byte) 0x40, (byte) 0x2b, 0x0f, 0x2d, (byte) 0x88, 0x55, 0x0f};
//            byte[] decrypted2 = des.des_de(mykey2, mykey2, "00000000".getBytes());
//            System.err.println(myDecrypt("A6402B0F2D88550F50D5062B5C8E563F46779299E05A8738A2287FCEE3B6AEF27DAD8CF0D45E9E74EB99769487FE278E", "00000000"));
//
////            byte[] num = {1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,};
////            System.out.println(des.bitToInt(num,32)[0]);
//            byte chiperhex1[] = {(byte) 0x6d, (byte) 0xa3, (byte) 0x7b, (byte) 0xe5, (byte) 0xd4, (byte) 0xb2, (byte) 0xef, (byte) 0xac};
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

    public static String myDecrypt(String hexStr, String key) {
        String result = "";
        Scratch des = new Scratch();
        int mod = (hexStr.length() >> 1) % 8;
        String notEncrypted = hexStr.substring(0, mod * 2);
        result += notEncrypted;
        String encrypted = hexStr.substring(mod * 2);
        int numOf8Bytes = encrypted.length() >> 4;
        for (int i = 0; i < numOf8Bytes; i++) {
            String this8Byte = encrypted.substring(i * 16, i * 16 + 16);
            byte[] nothing = new byte[8];
            byte[] decrypted = des.des_de(nothing, hexStrToBytes(this8Byte), key.getBytes());
            result += bytesToHexStr(decrypted);
        }
        return result;

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
