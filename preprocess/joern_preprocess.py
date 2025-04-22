import subprocess

def generate_JOERN(path,folder_path,repre):
    file_name = path.split("/")[-1].split(".")[0]
    out_path = folder_path + file_name

    command_parse = "../joern/joern-cli/joern-parse " + path
    command_export = "../joern/joern-cli/joern-export --repr "+repre+" --format dot --out " + out_path

    try:
        parse = subprocess.run(
            command_parse,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        result = subprocess.run(
            command_export,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        error = result.stderr
        return_code = result.returncode
        return return_code

    except Exception as e:
        print("Error create .bin file of:",path)
        print("Error message:",e)
        return None