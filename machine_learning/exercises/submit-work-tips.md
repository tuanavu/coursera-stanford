Mentor tips for submitting your work
---

[!Mentor tips for submitting your work](https://www.coursera.org/learn/machine-learning/discussions/vgCyrQoMEeWv5yIAC00Eog)

Tom MosherMentor · [5 months ago](https://www.coursera.org/learn/machine-learning/discussions/vgCyrQoMEeWv5yIAC00Eog) · Edited

This post is a collection of how-to's for fixing issues when you try to submit your work.

NOTE: This thread is closed to new comments because it has reached the maximum reply limit, and your new comments cannot be replied to.

If you use all the tips in this thread and still cannot submit your work - please start a new thread.

-------------------

A word about versions of Octave and MATLAB.

The On-Demand version of the ML course was tested using MATLAB R2015a and Octave 3.8.2. There is no guarantee that the programming assignment submit method will work with older versions.

------------------

**Before you attempt the first programming exercise**, watch all of the video lectures for Week 1 and Week 2.

--------------------------

The submit script has been modified since the videos were recorded. It no longer asks for what part you want to submit. Now it automatically evaluates any function that you have modified.

-------------------------------

**If your functions give the correct results **when you run the exercise script (ex1, etc...), but you get no credit when running the submit script, read this next paragraph carefully:

"The submit grader uses a different test case than the one in the exercise script. Your functions must work correctly with any size of data set. That includes the number of training samples, the number of features, and the number of labels."

-------------------------------

**Additional testing for your functions:**

- You can find test cases in the "Programming Exercise Test Cases" thread in the General Discussion area of the Forum.

**Tutorials:**

- You can find tutorials in the "Programming Exercise Tutorials" thread in the General Discussion area of the Forum. The tutorials cover the vectorized methods.

-------------------------------

**Debugging your code:**

You can debug inside your functions by placing a "keyboard" command in the code. When the code is executed, it will break to the debugger, where you can inspect the sizes of the variables, and try hand-entering some commands to find where the problem occurs.

-------------------------------

**If you are using MATLAB and see an error message like this.**..

_Submission failed: unexpected error: Invalid field name: 
```
'x0x_0x${sprintf('%X',unicode2native($))}__0x${sprintf('%X',unic'._
```

... it typically means you're using a version of MATLAB that is too old. Use the MATLAB version that is linked to this course (see Week 1 materials for instructions).

---------------------------------

**If you are using MATLAB and see an error like this in programming exercise 2...**

_Undefined function 'fminunc' for input arguments of type 'function_handle'_

...this means you do not have the Optimization Toolbox installed, or the license isn't activated. Search the MATLAB Help forum for assistance.

--------------------------------

**When you run the submit script, if you are seeing error messages that contain any of these phrases...**

**urlread, curl, urlreadwrite, peer certificate, CA certificate, unsupported protocol, JSONparser**

...here are some issues you can check.

- Are you using **Octave 4.0.0**? You will need to install this [&lt;patch&gt;](https://drive.google.com/file/d/0B6lXyE7fgSlXZjlqQ3FIRExmTDA/view?usp=sharing). Follow the instructions in the readme.txt file. Restart Octave after installing the patch. There is a bug in the printf() function, it will be fixed in Octave 4.0.1. The error "

JSONparser:invalidFormat: Outer level structure must be an object or an array" error is caused by this bug.

Post-installation test: cd to the 'lib' folder, enter this command: "makeValidFieldName('2')". Verify "ans = x0x32_".

- Are you getting an error about "**peer certificates**"? You will need to install [(this)](https://drive.google.com/file/d/0B6lXyE7fgSlXXy1nMXlpb3RyZ1E/view?usp=sharing) patch if you are using Windows, or [(this)](https://drive.google.com/file/d/0B6lXyE7fgSlXeEl6dG96SG5FcDA/view?usp=sharing) this patch if you are using a linux-based operating system (including MacOS). Follow the instructions in the readme.txt file. Restart Octave after installing the patch.
- Be sure you have write permission to the folder where the exercise scripts were extracted.
- The submit script wants your 

Coursera email address and the 

exercise Token for that exercise when you submit the work - NOT your Coursera password. This is especially true if you have the "urlreadwrite" error message. Each programming assignment page displays the Token for use with that exercise.

- Extract the contents of the exercise archive preserving their folder attributes. You should have an "ex?\" working folder, and subfolders "ex?\lib" and "ex?\lib\jsonlab". Do not change any of the folder names included in the exercise zip files.

-------------------------

**If you're using Ubuntu** and MATLAB R2014a, some students have reported that they can submit their work using Ubuntu 12.04 LTS, but not using Ubuntu 14.04.

----------------------

**If you run the submit script and get an error about "Undefined function or variable 'submitWithConfiguration':**

- Set your Octave or Matlab working directory (pwd) to the "..\machine-learning-ex?\ex?" folder where you extracted the exercise files. Do **

NOT** use addpath() to point to the exercise folder.

- Try getting a new Token
- Try using the Octave command-line version instead of the GUI.

-----------------------

If you see an error message like: **JSONparser:invalidFormat: Outer level structure must be an object or an array**

- Be sure you are using the correct email address, and that you haven't mis-typed it.
- Try getting a new Token. Note that there is a different Token for each programming exercise. You get the Token from the same page where you downloaded the programming exercise zip file.
- Try deleting any "token.mat" file from your "..\ex?" exercise folder, then run submit again and re-enter your email and Token.
- Be sure you copy-and-paste the Token, because the font has some similar-looking upper and lower-case letters.

**Other tips**

- Do you see this error "username or password cannot be verified"? The solution is to use only the exercise archive scripts from the On Demand course - not any previous version.

- Your cost functions should return the grad as a column vector - size (n x 1). The submit grader doesn't check for this, but other functions will complain and throw warnings if your grad is a row vector (1 x n) - that's bad.
- Do NOT use the dot() or mtimes() functions to perform a dot product between two vectors or matrices. Instead, use the '*' math operator.
- Folder names should NOT have any embedded 'space' characters. Use '-' or '_' instead.

The submit scripts want to connect to www-origin.coursera.org on port 443. Any of the following items can cause a connection failure, so try...

- Temporarily disable your firewall (if applicable)
- Temporarily disable your anti-virus software (if applicable)
- Configure Octave or Matlab to use your proxy settings (if applicable)

----------------------------------

There is a unique Token for each exercise, which is found on the download page for that exercise. There is also a link on that page to generate a new Token, which is worth trying when all else fails.

Sometimes deleting your "token.mat" file and re-running the submit script (re-entering your email address and Token) will fix a "...non-existent field 'email_address'. " runtime error.

Deleting your "token.mat" file is especially important if you are using both Octave and Matlab on the same computer.

--------------------------------

Special tip for ex6 - error in visualizeBoundary.m. [(see this tip to fix it)](https://www.coursera.org/learn/machine-learning/discussions/YTIKWMpuEeSWEiIAC0wC5g). This can cause either the 'hggroup' error message, or the decision boundary to not be displayed.

Special tip for ex8 - error in ex8_cofi.m [(see this tip to fix it)](https://www.coursera.org/learn/machine-learning/discussions/YI_8-NrxEeSIcSIAC0EU3g/replies/6TmTueGnEeSgrSIAC1ALYw)

--------------------------------

**After you've checked all of these items,** if you are still having a problem, then start a new discussion thread detailing your issue.

===============

keywords: mentor tips submit error submit problem submission error