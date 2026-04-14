def test_hyp3_glacier_melt(script_runner):
    ret = script_runner.run(['python', '-m', 'hyp3_glacier_melt', '-h'])
    assert ret.success
