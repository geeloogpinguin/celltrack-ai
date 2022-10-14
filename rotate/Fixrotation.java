import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.io.*;
import java.util.*;
import java.nio.file.*;
import javax.imageio.*;

public class Fixrotation {

	public static void main (String [] args) {
		new Fixrotation (args);
	}
	
	public Fixrotation (String [] args) {
		run (args);
	}
	
	private void run (String [] args) {
		if (args.length != 2) {
			System.err.println ("Usage: Fixrotation <im0> <im1>");
			System.exit (1);
		}
		File f0 = new File (args [0]);
		File f1 = new File (args [1]);
		
		
		// get extention of f1
		String [] sp = args [1].split ("\\.");
		String ext = sp [sp.length - 1];
		
		
		BufferedImage im0 = readImage (f0);
		BufferedImage im1 = readImage (f1);
		
		
		int [][] mx0 = image2mx (im0);
		//dumpMx (mx0);
		
		
		int n = 360;
		int ival = 1;
		System.out.println ("i\trad\tdeg\tdist");
		double mindist = Double.MAX_VALUE;
		int mini = -1;
		for (int i = 0; i < n; i += ival) {
			//double rad = (double) i / n * (360 / n) * Math.PI;
			double rad = i * 2 * Math.PI / n;
			int deg = i * 360 / n;
			BufferedImage im1rot = rotate (im1, rad);
			BufferedImage im1resized = resize (im1rot, im0.getWidth (), im0.getHeight ());
			//writeImage (im1resized, "jpg", new File ("fin-" + i + ".jpg"));
			int [][] mx1 = image2mx (im1resized);
			
			double dist = distance (mx0, mx1);
			if (dist < mindist) {
				mindist = dist;
				mini = i;
			}
			System.out.println (i + "\t" + rad + "  \t" + deg + "\t" + dist);
		}
		
		
		System.out.println ("best rotation = " + (mini * 360 / n) + " deg");
		
		// Rotate im1 with best rotation and store image
		BufferedImage fin = rotate (im1, (mini * 2 * Math.PI / n));
//		writeImage (fin, "jpg", new File ("best_rot.jpg"));
		writeImage (fin, ext, new File ("best_rot." + ext));
		
	}
	private BufferedImage readImage (File f) {
		// read image file and return BufferedImage
		try {
			System.out.println ("readImage (" + f + ")");
			BufferedImage img = ImageIO.read (f);
			if (img == null) {
				System.err.println ("img == null");
				System.exit (1);
			}
			return img;
		}
		catch (IOException x) {
			x.printStackTrace ();
		}
		return null;
	}
	
	private int numComponents (BufferedImage bi) {
		int nComp = 0;
		if (bi.getType () == BufferedImage.TYPE_3BYTE_BGR) {
			nComp = 3;
		}
		else if (bi.getType () == BufferedImage.TYPE_BYTE_GRAY) {
			nComp = 1;
		}
		else {
			System.out.println ("FixRotation :: image2mx :: unsupported image type: " + bi.getType ());
			System.out.println ("see https://docs.oracle.com/javase/7/docs/api/constant-values.html#java.awt.image.BufferedImage.TYPE_INT_RGB");
			System.out.println ("and add support for this image type");
			System.exit (1);
		}
		return nComp;
	}
	
	private int [][] image2mx (BufferedImage bi) {
		// convert BufferedImage to matrix (for RGB image get max(RGB) for each pixel)
		byte [] pix = ((DataBufferByte) bi.getRaster ().getDataBuffer ()).getData ();
		int [][] mx = new int [bi.getWidth ()][bi.getHeight ()];
		
		int nComp = numComponents (bi);
		if (nComp == 1) {
			for (int i = 0; i < pix.length; i++) {
				
				int val;
				if (pix [i] < 0) {
					//System.out.println ("Fixrotation :: image2mx :: negative value: pix [i] = " + pix [i]);
					val = 255 + pix [i];
				}
				else {
					val = pix [i];
				}
				
				int x = i % bi.getWidth ();
				int y = i / bi.getWidth ();
				mx [x][y] = val;
			}
		}
		else if (nComp == 3) {
			for (int i = 0; i < (bi.getWidth () * bi.getHeight ()); i++) {
				int val = Math.max (pix [i * 3], Math.max (pix [i * 3 + 1], pix [i * 3 + 2]));
				
				int x = i % bi.getWidth ();
				int y = i / bi.getWidth ();
				mx [x][y] = val;
			}
		}
		
		return mx;
	}
	
	private static void dumpMx (int [][] mx) {
		String seq = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
		int h = mx [0].length;
		int w = mx.length;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int index = mx [x][y] * (seq.length () - 1) / 255;
				System.out.print (seq.charAt (index));
			}
			System.out.println ();
		}
	}
	
	private BufferedImage rotate (BufferedImage image, double angle) { // angle in rad [0 .. 2*pi]
		double sin = Math.abs (Math.sin (angle));
		double cos = Math.abs (Math.cos (angle));
		int w = image.getWidth ();
		int h = image.getHeight ();
		int neww = (int) Math.floor (w * cos + h * sin);
		int newh = (int) Math.floor(h * cos + w * sin);
		
		BufferedImage result = new BufferedImage (neww, newh, image.getType ());
		Graphics2D g = result.createGraphics ();
		g.translate ((neww - w) / 2, (newh - h) / 2);
		g.rotate (angle, w / 2, h / 2);
		g.drawRenderedImage (image, null);
		g.dispose ();
		return result;
	}
	
	private static BufferedImage deepCopy (BufferedImage bi) {
		ColorModel cm = bi.getColorModel ();
		boolean isAlphaPremultiplied = cm.isAlphaPremultiplied ();
		WritableRaster raster = bi.copyData (null);
		return new BufferedImage (cm, raster, isAlphaPremultiplied, null);
	}
	
	private BufferedImage mySubimage (BufferedImage src, int x, int y, int w, int h) {
		// Enhanced version of BufferedImage.getSubimage
		int [][] mxsrc = image2mx (src);
		byte [] data = new byte [w * h];
		
		int i = 0;
		for (int sy = 0; sy < h; sy++) {
			for (int sx = 0; sx < w; sx++) {
				data [i] = (byte) mxsrc [x + sx][y + sy];
				i++;
			}
		}
		
		BufferedImage sub2 = new BufferedImage (w, h, BufferedImage.TYPE_BYTE_GRAY);
		byte [] a = ((DataBufferByte) sub2.getRaster ().getDataBuffer ()).getData ();
		System.arraycopy (data, 0, a, 0, data.length);
		return sub2;
	}
	
	private BufferedImage resize (BufferedImage src, int nw, int nh) {
		// resize src to nw * nh
		int ow = src.getWidth ();
		int oh = src.getHeight ();
		
		//System.out.println ("WIDTH:");
		//System.out.println ("  old: " + ow);
		//System.out.println ("  new: " + nw);
		
		//System.out.println ("HEIGHT:");
		//System.out.println ("  old: " + oh);
		//System.out.println ("  new: " + nh);
		
		BufferedImage cpy = deepCopy (src);
		if (nw > ow || nh > oh) {
			// First enlarge, then crop
			
			int maxw = Math.max (nw, ow);
			int maxh = Math.max (nh, oh);
			//System.out.println ("maxw = " + maxw + " maxh = " + maxh);
			
			int xoffset = (maxw - ow) / 2;
			int yoffset = (maxh - oh) / 2;
			//System.out.println ("xoffset = " + xoffset);
			//System.out.println ("yoffset = " + yoffset);
			
			BufferedImage trg = new BufferedImage (maxw, maxh, src.getType ());
			trg.createGraphics ().drawImage (cpy, xoffset, yoffset, null);
			
			int xo = (maxw - nw) / 2;
			int yo = (maxh - nh) / 2;
			BufferedImage trg2 = mySubimage (trg, xo, yo, nw, nh);
			return trg2;
		}
		else {
			// Crop directly
			int xo = (ow - nw) / 2;
			int yo = (oh - nh) / 2;
			
			BufferedImage trg = mySubimage (cpy, xo, yo, nw, nh);
			return trg;
		}
	}
	
	private void writeImage (BufferedImage im, String type, File f) {
		try {
			ImageIO.write (im, type, f);
		}
		catch (IOException x) {
			x.printStackTrace ();
		}
		
	}
	
	
	private double distance (int [][] mx0, int [][] mx1) {
		if (mx0.length != mx1.length || mx0.length < 1 || mx0 [0].length != mx1 [0].length) {
			System.err.println ("mx0 and mx1 are not equal sized");
			System.exit (1);
		}
		double dist = 0;
		int w = mx0.length;
		int h = mx0 [0].length;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				double err = Math.sqrt ((mx0 [x][y] - mx1 [x][y]) * (mx0 [x][y] - mx1 [x][y]));
				dist += err;
			}
		}
		return dist;
	}
	
/*
	private BufferedImage deeperCopy (BufferedImage bi) {
		//BufferedImage trg = new BufferedImage (bi.getWidth (), bi.getHeight (), bi.getType ());
		//byte [] pix = ((DataBufferByte) bi.getRaster ().getDataBuffer ()).getData ();
		
		try {
			ByteArrayOutputStream baos = new ByteArrayOutputStream ();
			ImageIO.write (bi, "jpg", baos);
			baos.flush ();
			
			InputStream in = new ByteArrayInputStream (baos.toByteArray ());
			baos.close ();
			
			BufferedImage trg = ImageIO.read (in);
			
			return trg;
		}
		catch (IOException x) {
			x.printStackTrace ();
		}
		return null;
	}
	
	
*/
}


