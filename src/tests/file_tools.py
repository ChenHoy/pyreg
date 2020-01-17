"""
This tool file contains several file manipulation methods
"""

import os
from PyPDF2 import PdfFileMerger


def merge_pdfs(script_dir, input_dir, title):
  """
  Merge pdfs in a directory into one document

  args:
    script_dir (str): cwd
    input_dir (str): Target directory
    title (str): Title for report
  """
  abs_file_path = os.path.join(script_dir, input_dir)
  try:
    os.chdir(abs_file_path)
    files = []
    for f_in in os.listdir():
      if f_in.endswith(".pdf"):
        files.append(f_in)
    files.sort()   
    #~ print(files) 
    merger = PdfFileMerger()
    for pdf in files:
      merger.append(open(pdf, 'rb'))
    with open(title, "wb") as f_out:
      merger.write(f_out)
  finally:
    os.chdir(script_dir)
  
  return 


def make_gif(framedelay, script_dir, fig_path, pgf_file_name):
  """ Convert the pdf report into a gif with ImageMagick """ 
  abs_file_path = os.path.join(script_dir, fig_path)
  try:
    os.chdir(abs_file_path)
    os.system("convert -verbose -delay " + str(framedelay) + " -loop 0 -density 400 " + str(pgf_file_name) + " animation.gif")  
  finally:
    os.chdir(script_dir)
  
  return

def make_mp4(script_dir, fig_path):
  abs_file_path = os.path.join(script_dir, fig_path)
  try:
    os.chdir(abs_file_path)
    os.system("ffmpeg -f gif -i animation.gif animation.mp4")
  finally:
    os.chdir(script_dir)
    
  return

